#

"""The code to accompany the paper 'A. Ringh, X. Zhuge, W. J. Palenstijn,
K. J. Batenburg, and O. Öktem. High-level algorithm prototyping: an example
extending the TVR-DART algorithm'. All reconstructions in the paper can be
obtained by running the script with corresponding parameters."""

import odl
import numpy as np
import scipy
import time
import sys
import skimage.measure

# The implementations needed for the TVR-DART algorithm
import dart_ops
from huber_func import HuberNorm

# Originally odl files, but that contains minor changes.
import steplen
import default_functionals
import tensor_ops


# Seed randomness for reproducability
np.random.seed(seed=10)

# Pixel sizes
num_pixels_pad = 32  # Pads the phantom with zero pixels
num_pixels = 256  # Number of pixels in the phantom

# This computes how many pixels the detector should be in order to have one bin
# for each ray also when sampling from 45 degrees.
num_detector_pixels = 50 * int(np.ceil(np.sqrt(2)*(num_pixels + 2*num_pixels_pad)/50))

# Discrete reconstruction space: discretized functions on the rectangle
reco_space = odl.uniform_discr(min_pt=[-200, -200], max_pt=[200, 200],
                               shape=[num_pixels+2*num_pixels_pad,
                                      num_pixels+2*num_pixels_pad],
                               dtype='float64')

# Make a parallel beam geometry with flat detector
angle_partition = odl.uniform_partition(0, np.pi, 18, nodes_on_bdry=True)
detector_partition = odl.uniform_partition(-200, 200, num_detector_pixels)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Ray transform (= forward projection). We use ASTRA CUDA backend.

ray_trafo_unscaled = odl.tomo.RayTransform(reco_space, geometry,
                                           impl='astra_cuda')

directory = '/home/aringh/Documents/TVR-DART/Reconstructions/'

# =============================================================================
# Parameter selection
# =============================================================================

# Select the phantom. Currently supporeted: 'derenzo' and 'phantom3'
phantom_name = 'phantom3'

# Select which noise model is to be used. Supported are 'no_noise', 'poisson',
# 'white', and 'poisson_and_white'
noise_model = 'poisson'

# Set values that corresponds to noise. Only the parameters that are relevant
# for the selected noise model will have an effect.
dose = 1.0  # Corresponds to exposure time, and affects Poisson noise.
white_noise_level = 0.05

# Base value for the soft thresholding function
base_value = 0.0

# If the TVR-DART optimization should be over parameters as well
optimize_parameters = True

# Data functional. Supported are 'l2' and 'kl'
data_mathcing = 'kl'

# Which initiall guess to use. Supporeted are 'fbp' and 'cg_normal'
initial_guess_method = 'fbp'
fbp_filter_param = 0.15

# The regularization parameters
reg_param = 1.0
reg_param_TV = 0.1


# =============================================================================
# Create strings for saving
# =============================================================================
save_dose = str(dose).replace('.', '_')
save_white = str(white_noise_level).replace('.', '_')
save_reg_param = str(reg_param).replace('.', '_')
save_reg_param_TV = str(reg_param_TV).replace('.', '_')
no_title = '__no_title'

time_now = time.strftime("%Y_%m_%d__%H_%M_%S")

output_filename = ('Terminal_output' + '_phantom_' + phantom_name + '_noise_' +
                   noise_model + '_' + time_now + '.txt')


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(directory + '/' + output_filename)


# =============================================================================
# Create the phantom. Currently supporeted: 'derenzo' and 'phantom3'
# =============================================================================

if phantom_name == 'derenzo':
    phantom = odl.phantom.derenzo_sources(reco_space)

    thresholds = [0.6]
    values = [1.1]
    sharpness = [6]

    # "Correct values"
#    thresholds = [0.5]
#    values = [1.0]
#    sharpness = [6]

elif phantom_name == 'phantom3':
    image = np.rot90(scipy.misc.imread(
        'phantom_3.png'), -1)
    image = np.pad(array=image, pad_width=num_pixels_pad, mode='constant')
    phantom = reco_space.element(image)/255

    thresholds = [0.2, 0.8]
    values = [0.4, 1.1]
    sharpness = [10, 10]

    # "Correct values"
#    thresholds = [0.25, 0.75]
#    values = [0.502, 1.0]
#    sharpness = [10, 10]

else:
    raise ValueError('No phantom with that name')

phantom.show('The phantom')

# =============================================================================
# Create data
# =============================================================================
ray_trafo = dose * ray_trafo_unscaled

noise_free_data = ray_trafo(phantom)
noise_free_data.show('Noise free data')


if noise_model == 'no_noise':
    data_raw = noise_free_data.copy()

elif noise_model == 'poisson':
    # The noise level is goverened indirectly by the dose parameter
    data_raw = odl.phantom.poisson_noise(noise_free_data)

elif noise_model == 'white':
    # Additive white noise
    noise = odl.phantom.white_noise(ray_trafo.range)
    noise = noise * 1/noise.norm() * noise_free_data.norm() * white_noise_level
    data_raw = noise_free_data.copy() + noise

elif noise_model == 'poisson_and_white':
    # First Poisson noise, then adds white noise to the data
    data_raw = odl.phantom.poisson_noise(noise_free_data.copy())

    noise = odl.phantom.white_noise(ray_trafo.range)
    noise = noise * 1/noise.norm() * noise_free_data.norm() * white_noise_level
    data_raw = data_raw + noise

else:
    raise ValueError('No noise model with that name')


noise = data_raw - noise_free_data
print('Noise level, (data - noise_free_data).norm()/noise_free_data.norm(): {}'
      ''.format((data_raw - noise_free_data).norm()/noise_free_data.norm()))

data_raw.show('Data')


# Setting up for the TV-functional
gradient = odl.Gradient(reco_space)

pnorm = tensor_ops.PointwiseNorm(gradient.range)
tv_func = HuberNorm(reco_space, 0.0001) * pnorm * gradient


# =============================================================================
# Select which data matching term to use.
# =============================================================================
if data_mathcing == 'l2':
    data = data_raw.copy()
    data_func = (odl.solvers.L2NormSquared(ray_trafo.range).translated(data) *
                 ray_trafo)

elif data_mathcing == 'kl':
    tmp = data_raw.copy().asarray()
    less_than_zero_indices = tmp < 0
    tmp[less_than_zero_indices] = 0
    data = ray_trafo_unscaled.range.element(tmp)

    data_func = (default_functionals.KullbackLeibler(ray_trafo.range, prior=data) *
                 ray_trafo)
else:
    raise ValueError('No data matching term with that name')


# =============================================================================
#  Select initial guess.
# =============================================================================

# Create an fbp operator in order to compare to later
fbp_op = odl.tomo.fbp_op(ray_trafo_unscaled, filter_type='Hann',
                         frequency_scaling=fbp_filter_param)


if initial_guess_method == 'fbp':
    # Scaling of data, since fbp assumes unscaled operator
    # (benign modification)
    x = fbp_op(data/dose)
    x.show('FBP, also used as initial guess', saveto=(directory + '/' + phantom_name + '_fbp__noise_' +
                    noise_model + '__ dose_' +
                    save_dose + '__white_' + save_white))
    x.show(saveto=(directory + '/' + phantom_name + '_fbp__noise_' +
                   noise_model + '__ dose_' + save_dose + '__white_' +
                   save_white))

elif initial_guess_method == 'cg_normal':
    x = reco_space.zero()
    odl.solvers.conjugate_gradient_normal(ray_trafo, x, data, niter=5)
    x.show('Initial guess base on gcn', saveto=(directory + '/' + phantom_name + '_cgn__noise_' +
                    noise_model + '__ dose_' +
                    save_dose + '__white_' + save_white))
    x.show(saveto=(directory + '/' + phantom_name + '_cgn__noise_' +
                   noise_model + '__ dose_' + save_dose + '__white_' +
                   save_white))

else:
    raise ValueError('No initial guess with that name')

# Print figure of mertis
rme_initial = np.abs(x.asarray() - phantom.asarray()).sum() / phantom.asarray().sum()
ssim_initial = skimage.measure.compare_ssim(x.asarray(), phantom.asarray())
print('FOMs for initial guess reconstruction:')
print('-- RME: {}'.format(rme_initial))
print('-- SSIM: {}'.format(ssim_initial))

# =============================================================================
# Set regularization parameter
# =============================================================================

# Create a vector in rn with the parameters
x_para = thresholds.copy() + values.copy()
rn_para = odl.rn(len(x_para))
x_para = rn_para.element(x_para)

# Create the soft thresholding operator
threshold_op = dart_ops.SoftThresholdOperator(reco_space, base_value,
                                              thresholds, values, sharpness)


# Things in order to display intermediat results, without opening to many
# new figures
class op_used_in_show():
    def __init__(self, op):
        self.__op = op

    @property
    def op(self):
        return self.__op

    @op.setter
    def op(self, new_op):
        self.__op = new_op

    def __call__(self, x):
        return self.op(x)

# Used to display intermediate iterates
shower = odl.solvers.CallbackShow(step=1)
show_heper = op_used_in_show(threshold_op)
callback_image = (odl.solvers.CallbackPrintIteration() &
                  (lambda x: shower(show_heper(x))))

callback_param = odl.solvers.CallbackPrintIteration()

# Used in order to debug
# np.seterr(invalid='raise')


if optimize_parameters:
    # Run the algorithm: alternating optimization/coordinate descent
    for i in range(10):
        ###########################################################################
        # This part of the iteration optimize over the reconstruction/image
        ###########################################################################
        print('Optimizing over the image, iteration {}'.format(i))

        # The optimization functional when optimizing over the image
        f = data_func * threshold_op + reg_param * tv_func * threshold_op

        # Line-search
        linesearch = steplen.BacktrackingLineSearch(f)  # , max_num_iter=200)

        # Solve the problem using BFGS
        callback_image.reset()
        odl.solvers.bfgs_method(f=f, x=x, line_search=linesearch, maxiter=2,
                                tol=1e-8, callback=callback_image, num_store=None)

        ###########################################################################
        # This part of the iteration optimize over the parameters
        ###########################################################################
        print('Optimizing over the parameters, iteration {}'.format(i))

        # Update thresholod values and then operators here
        para_op = dart_ops.SoftThresholdReducedParamOperator(
            x=x, base_value=base_value, num_thresholds=len(thresholds),
            num_values=len(values), sharpness=sharpness)

        # The optimization functional when optimizing over the parameters
        f = data_func * para_op + reg_param * tv_func * para_op

        # Line-search
        linesearch = steplen.BacktrackingLineSearch(f)

        # Reset the callback that prints iteration number
        callback_param.reset()

        # Solve the problem using BFGS
        print('x_para before update: {}'.format(x_para))
        odl.solvers.bfgs_method(f=f, x=x_para, line_search=linesearch, maxiter=10,
                                tol=1e-8, callback=callback_param, num_store=None)
        print('x_para after update: {}'.format(x_para))

        # Store the new values
        thresholds = x_para[0:len(thresholds)]
        values = x_para[len(thresholds):]

        threshold_op = dart_ops.SoftThresholdOperator(reco_space, base_value,
                                                      thresholds, values,
                                                      sharpness)
        show_heper.op = threshold_op
        callback_image.reset()
else:
    # The optimization functional
    f = data_func * threshold_op + reg_param * tv_func * threshold_op

    # Line-search
    linesearch = steplen.BacktrackingLineSearch(f)

    # Solve the problem using BFGS
    callback_image.reset()
    odl.solvers.bfgs_method(f=f, x=x, line_search=linesearch, maxiter=500,
                            tol=1e-6, callback=callback_image, num_store=20)

# Display the final result
x_reco = threshold_op(x)
x_reco.show('TVR-DART reconstruction',
            saveto=(directory + '/' + phantom_name + '_tvr-dart__noise_' +
                    noise_model + '__ dose_' +
                    save_dose + '__white_' + save_white + '__functional_' +
                    data_mathcing + '__reg_' + save_reg_param))
x_reco.show(saveto=(directory + '/' + phantom_name + '_tvr-dart__noise_' +
                    noise_model + '__ dose_' +
                    save_dose + '__white_' + save_white + '__functional_' +
                    data_mathcing + '__reg_' + save_reg_param + no_title))
np.save((directory + '/' + phantom_name + '_tvr-dart__noise_' +
         noise_model + '__ dose_' + save_dose +
         '__white_' + save_white + '__functional_' + data_mathcing + '__reg_' +
         save_reg_param), np.asarray(x_reco))

print('Final values for thresholds: {}'.format(thresholds))
print('Final values for values: {}'.format(values))
print('Final values for sharpness: {}'.format(sharpness))

# Print figure of mertis
rme = np.abs(x_reco.asarray() - phantom.asarray()).sum() / phantom.asarray().sum()
ssim = skimage.measure.compare_ssim(x_reco.asarray(), phantom.asarray())
print('FOMs for TVR-DART reconstruction:')
print('-- RME: {}'.format(rme))
print('-- SSIM: {}'.format(ssim))


# =============================================================================
# Make a TV-reconstruction
# =============================================================================
grad = odl.Gradient(reco_space)

f = odl.solvers.functional.IndicatorBox(reco_space, lower=float(base_value),
                                        upper=float(values[-1]))

# Use the same data functional for TV as for TVR-DART
if data_mathcing == 'l2':
    data_func = odl.solvers.L2NormSquared(ray_trafo.range).translated(data)

elif data_mathcing == 'kl':
    data_func = default_functionals.KullbackLeibler(ray_trafo.range, prior=data)

else:
    raise ValueError('No data matching term with that name for TV')

l1_TV = reg_param_TV * odl.solvers.GroupL1Norm(grad.range)

lin_ops = [ray_trafo, grad]
g = [data_func, l1_TV]

ray_norm = odl.power_method_opnorm(ray_trafo)
grad_norm = odl.power_method_opnorm(grad)

sigma = [1 / ray_norm**2, 1 / grad_norm**2]
tau = 1.0

callback = (odl.solvers.CallbackPrintIteration())

x_tv = reco_space.one()

odl.solvers.douglas_rachford_pd(x=x_tv, f=f, g=g, L=lin_ops, tau=tau,
                                sigma=sigma, niter=2000, callback=callback)

x_tv.show('TV-reconstruction', saveto=(directory + '/' + phantom_name +
                                       '_TV__noise_' +
                                       noise_model + '__ dose_' + save_dose +
                                       '__white_' + save_white +
                                       '__functional_' + data_mathcing +
                                       '__reg_' + save_reg_param_TV))

x_tv.show(saveto=(directory + '/' + phantom_name + '_TV__noise_' +
                  noise_model + '__ dose_' + save_dose +
                  '__white_' + save_white + '__functional_' + data_mathcing +
                  '__reg_' + save_reg_param_TV + no_title))

np.save((directory + '/' + phantom_name + '_TV__noise_' +
         noise_model + '__ dose_' + save_dose +
         '__white_' + save_white + '__functional_' + data_mathcing +
         '__reg_' + save_reg_param_TV), np.asarray(x_tv))

# Print figure of mertis
rme_tv = np.abs(x_tv.asarray() - phantom.asarray()).sum() / phantom.asarray().sum()
ssim_tv = skimage.measure.compare_ssim(x_tv.asarray(), phantom.asarray())
print('FOMs for TV reconstruction:')
print('-- RME: {}'.format(rme_tv))
print('-- SSIM: {}'.format(ssim_tv))


# =============================================================================
# Show the thresholding operator
# =============================================================================
my_space = odl.uniform_discr(0, 1, 1000)
my_x = my_space.element(lambda x: x)
my_op = threshold_op = dart_ops.SoftThresholdOperator(my_space, base_value,
                                                      thresholds, values,
                                                      sharpness)
my_op(my_x).show()

sys.stdout.log.close
sys.stdout = sys.stdout.terminal
