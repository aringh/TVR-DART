#

""" An example implementation of the DART algorithm."""

import odl
import numpy as np
import scipy

import dart_ops as DART_stuff


###############################################################################
# Set up the inverse problem
###############################################################################
# Discrete reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[256, 256], dtype='float64')

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 18, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, 18)
# Detector: uniformly sampled, n = 450, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 450)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Ray transform (= forward projection). We use ASTRA CUDA backend.
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

# Create a phantom
image = np.rot90(scipy.misc.imread(
        '/home/aringh/git/odl-private/odl/solvers/discrete/phantom_3.png'), -1)
phantom = reco_space.element(image)/255

# Create edge-detection and threshold operators
thresholds = [0.25, 0.75]
values = [0, 0.5, 1]

threshold_op = DART_stuff.ThresholdOperator(reco_space, thresholds, values)
edge_op = DART_stuff.EdgeDetectOperator(reco_space, diagonal_neighbour=False)

# Initialize convolution operator by Fourier formula
#     conv(f, g) = F^{-1}[F[f] * F[g]]
# Where F[.] is the Fourier transform and the fourier transform of a guassian
# with standard deviation filter_width is another gaussian with width
# 1 / filter_width
filter_width = 1.0  # standard deviation of the Gaussian filter
ft = odl.trafos.FourierTransform(reco_space)
c = filter_width ** 2 / 4.0 ** 2
gaussian = ft.range.element(lambda x: np.exp(-(x[0] ** 2 + x[1] ** 2) * c))
convolution = ft.inverse * gaussian * ft


# Create data
data = ray_trafo(phantom)
data += odl.phantom.white_noise(ray_trafo.range) * np.mean(data) * 0.1


###############################################################################
# DART-reconstruciton
###############################################################################
# Create a starting guess
x = reco_space.one()
odl.solvers.conjugate_gradient_normal(ray_trafo, x, data, niter=5)

x.show('Intial reconstruction guess')

x_thresholded = threshold_op(x)
x_thresholded.show('Initial thresholded guess')

callback1 = odl.solvers.CallbackShow()
callback2 = odl.solvers.CallbackShow()


for i in range(50):
    print('Itertion {}'.format(i))
    x_edge = edge_op(x_thresholded)
    callback1(x_edge)

    # Random part of DART
    x_edge = np.maximum(x_edge,
                        np.float32(np.random.uniform(size=reco_space.shape) >
                                   0.95))

    free_op = odl.MultiplyOperator(x_edge)
    fix_op = odl.MultiplyOperator(1-x_edge)

    free_data = data - (ray_trafo * fix_op)(x_thresholded)

    op = ray_trafo * free_op

    x_tmp = free_op(x)
    odl.solvers.conjugate_gradient_normal(op, x_tmp, free_data, niter=30)

    x_tmp += fix_op(x_thresholded)

    x = convolution(x_tmp)
    x_thresholded = threshold_op(x)
    callback2(x_thresholded)

x_thresholded.show('Final DART reconstruction')


###############################################################################
# TV-reconstruction
###############################################################################
grad = odl.Gradient(reco_space)

f = odl.solvers.functional.IndicatorBox(reco_space, 0)

g_l2sq = odl.solvers.L2NormSquared(ray_trafo.range).translated(data)
g_l1 = 0.3 * odl.solvers.L1Norm(grad.range)

lin_ops = [ray_trafo, grad]
g = [g_l2sq, g_l1]

ray_norm = odl.power_method_opnorm(ray_trafo)
grad_norm = odl.power_method_opnorm(grad)

sigma = [1 / ray_norm**2, 1 / grad_norm**2]
tau = 1.0

callback = (odl.solvers.CallbackPrintIteration())

x_tv = reco_space.zero()

odl.solvers.douglas_rachford_pd(x=x_tv, f=f, g=g, L=lin_ops, tau=tau,
                                sigma=sigma, niter=2000, callback=callback)

x_tv.show('TV-reconstruction with nonnegativity')
