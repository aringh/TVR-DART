#

"""Operators used in the DART and TVR-DART algorithms."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

from odl.operator.operator import Operator
from odl.operator.default_ops import MultiplyOperator
from odl.space.space_utils import rn
from odl.discr.lp_discr import DiscreteLp
from odl.discr.diff_ops import Gradient

import numpy as np


# TODO: implement for arbitrary soft threshold functions. Take them as a list?
class SoftThresholdOperator(Operator):

    """Soft thresholding operator.

    This operator is mapping from image space to image space, and the values
    defining the thresholds are seen as fixed parameters.
    """

    def __init__(self, domain, base_value, thresholds, values, sharpness):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `DiscreteLp` or `FnBase`
            Domain of the operator.
        base_value : float
            The lowest value of threshold to.
        thresholds : array of float
            The threshold/mid-point values.
        values : array of float
            The gray-scale values values.
        sharpness : array of float
            The sharpness values.
        """
        if not len(thresholds) == len(values):
            raise ValueError('`thresholds` {} needs to be same length as '
                             '`values` {}'.format(thresholds, values))

        if not len(thresholds) == len(sharpness):
            raise ValueError('`thresholds` {} needs to be same length as '
                             '`sharpness` {}'.format(thresholds, values))

        if base_value > values[0]:
            raise ValueError('`base_value` {} cannot be larger than the first '
                             'value in `values` {}'.format(base_value, values))

        self.base_value = base_value
        self.thresholds = thresholds
        self.values = values
        self.sharpness = sharpness

        super().__init__(domain=domain, range=domain)

        # Create the differences between different threshold values
        self.delta_values = [self.values[0] - self.base_value]
        self.delta_values += [self.values[i] - self.values[i-1] for
                              i in range(1, len(self.values))]

        # Generate the k_g values
        self.k_g = [self.sharpness[i] / self.delta_values[i] for i in
                    range(len(self.sharpness))]

        self.exp_func = [self.factory_exp_fun(i) for i in self.k_g]

    def factory_exp_fun(self, k_g):
        """Helper function to create exponential functions."""
        def exp_fun(x):
            return np.exp(-2.0 * k_g * x)

        return exp_fun

    # TODO: Update to work for ProductSpace
    def _call(self, x):
        """Apply the soft thresholding operator to a point ``x``."""
        out = self.range.one() * self.base_value

        # TODO: remove this setting that ignores errors?
        # In order to aviod utprints in the terminal
        err_backup = np.seterr(over='ignore', under='ignore')

        for i in range(len(self.k_g)):
            out += (self.delta_values[i] *
                    1 / (1 + self.exp_func[i](x - self.thresholds[i])))

        np.seterr(**err_backup)

        return out

    def derivative(self, point):
        """Derivative of the operator."""
        # TODO: remove this setting that ignores errors?
        # In order to aviod outprints in the terminal
        err_backup = np.seterr(over='ignore', under='ignore')

        tmp = self.exp_func[0](np.abs(point - self.thresholds[0]))
        tmp = (2 * self.k_g[0] * tmp /
               (1 + tmp)**2)
        out = MultiplyOperator(self.delta_values[0] * tmp)
        for i in range(1, len(self.k_g)):
            tmp = self.exp_func[i](np.abs(point - self.thresholds[i]))
            tmp = (2 * self.k_g[i] *
                   tmp /
                   (1 + tmp)**2)
            out += MultiplyOperator(self.delta_values[i] * tmp)

        np.seterr(**err_backup)

        return out


# TODO: implement for arbitrary soft threshold functions. Take them as a list?
class SoftThresholdParamOperator(Operator):

    """Soft thresholding operator.

    This operator is mapping from parameter space to image space, and the
    image on which the soft thresholding is applied is seen as fixed parameter.
    """

    def __init__(self, x, base_value, num_thresholds, num_values,
                 num_sharpness):
        """Initialize a new instance.

        Parameters
        ----------
        x : element of ``DiscreteLp``-space
            The image
        base_value : float
            The lowest value of threshold to.
        num_thresholds : int
            The number of threshold/mid-point values that defines the operator.
        num_values : int
            The number of threshold gray-scale values that defines the
            operator.
        num_sharpness : int
            The number of sharpness values that defines the operator.
        """
        if not num_thresholds == num_values:
            raise ValueError('`num_thresholds` {} must be the same as '
                             '`num_values` {}'.format(num_thresholds,
                                                      num_values))

        if not num_thresholds == num_sharpness:
            raise ValueError('`num_thresholds` {} must be the same as '
                             '`num_sharpness` {}'.format(num_thresholds,
                                                         num_sharpness))

        self.base_value = base_value

        self.x = x

        self.num_thresholds = num_thresholds
        self.num_values = num_values
        self.num_sharpness = num_sharpness

        domain = rn(self.num_thresholds + self.num_values + self.num_sharpness)

        super().__init__(domain=domain, range=self.x.space)

    def exp_fun(self, x, k_g):
        """Helper function to create exponential functions."""
        return np.exp((-2.0 * k_g) * x)

    # TODO: Update to work for ProductSpace
    def _call(self, x):
        """Apply the soft thresholding operator to a point ``x``."""
        thresholds = x[0:self.num_thresholds]
        values = x[self.num_thresholds:self.num_thresholds+self.num_values]
        sharpness = x[self.num_thresholds+self.num_values:]

        # Create the differences between different threshold values
        delta_values = [values[0] - self.base_value]  # First value is fixed
        delta_values += [values[i] - values[i-1] for i in
                         range(1, self.num_values)]

        # Generate the k_g values
        k_g = [sharpness[i] / delta_values[i] for i in
               range(self.num_sharpness)]

        out = self.range.one() * self.base_value

        # TODO: remove this setting that ignores errors?
        # In order to aviod utprints in the terminal
        err_backup = np.seterr(over='ignore', under='ignore')

        for i in range(len(k_g)):
            out += (delta_values[i] *
                    1 / (1 + self.exp_fun(self.x - thresholds[i], k_g[i])))

        np.seterr(**err_backup)

        return out

    def derivative(self, point):
        """Derivative of the operator."""
        thresholds = point[0:self.num_thresholds]
        values = point[self.num_thresholds:self.num_thresholds+self.num_values]
        sharpness = point[self.num_thresholds+self.num_values:]

        # Create the differences between different threshold values
        delta_values = [values[0] - self.base_value]  # First value is fixed
        delta_values += [values[i] - values[i-1] for i in
                         range(1, self.num_values)]

        # Generate the k_g values
        k_g = [sharpness[i] / delta_values[i] for i in
               range(self.num_sharpness)]

        # Create the operator. Create the first component outside the loop
        # TODO: remove this setting that ignores errors?
        # In order to aviod utprints in the terminal
        err_backup = np.seterr(over='ignore', under='ignore')

        tmp = self.exp_fun(np.abs(self.x - thresholds[0]), k_g[0])
        ops = ((-2.0 * sharpness[0] *
                tmp /
                (1 + tmp)**2) *
               proj(self.domain, 0))

        for i in range(1, self.num_thresholds):
            tmp = self.exp_fun(np.abs(self.x - thresholds[i]), k_g[i])
            ops += ((-2.0 * sharpness[i] * tmp / (1 + tmp)**2) *
                    proj(self.domain, i))

        for i in range(self.num_values):
            tmp = self.exp_fun(self.x - thresholds[i], k_g[i])
            tmpABS = self.exp_fun(np.abs(self.x - thresholds[i]), k_g[i])
            ops += ((1 / (1 + tmp) - 2 * sharpness[i] *
                     (self.x - thresholds[i]) * tmpABS /
                     ((1 + tmpABS)**2 * delta_values[i])) *
                    proj(self.domain, self.num_thresholds + i))

        for i in range(self.num_sharpness):
            tmp = self.exp_fun(np.abs(self.x - thresholds[i]), k_g[i])
            ops += ((2 * (self.x - thresholds[i]) * tmp / (1 + tmp)**2) *
                    proj(self.domain,
                         self.num_thresholds + self.num_values + i))

        np.seterr(**err_backup)

        return ops


class SoftThresholdReducedParamOperator(Operator):

    """Soft thresholding operator.

    This operator is mapping from parameter space to image space, and the
    image on which the soft thresholding is applied is seen as fixed parameter.
    In this operator, also the sharpness values are seen as a fixed parameters.
    """

    def __init__(self, x, base_value, num_thresholds, num_values, sharpness):
        """Initialize a new instance.

        Parameters
        ----------
        x : element of ``DiscreteLp``-space
            The image
        base_value : float
            The lowest value of threshold to.
        num_thresholds : int
            The number of threshold/mid-point values that defines the operator.
        num_values : int
            The number of threshold gray-scale values that defines the
            operator.
        sharpness : array of float
            The sharpness values.
        """
        if not num_thresholds == num_values:
            raise ValueError('Update this')

        if not num_thresholds == len(sharpness):
            raise ValueError('Update this')

        self.x = x

        self.num_thresholds = num_thresholds
        self.num_values = num_values
        self.sharpness = sharpness

        self.base_value = base_value

        domain = rn(self.num_thresholds + self.num_values)

        super().__init__(domain=domain, range=self.x.space)

    def exp_fun(self, x, k_g):
        return np.exp((-2.0 * k_g) * x)

    # TODO: Update to work for ProductSpace
    def _call(self, x):
        """Apply the soft thresholding operator to a point ``x``."""
        thresholds = x[0:self.num_thresholds]
        values = x[self.num_thresholds:]

        # Create the differences between different threshold values
        delta_values = [values[0] - self.base_value]  # First value is fixed
        delta_values += [values[i] - values[i-1] for i in
                         range(1, self.num_values)]

        # Generate the k_g values
        k_g = [self.sharpness[i] / delta_values[i] for i in
               range(self.num_values)]

        out = self.range.one() * self.base_value
        err_backup = np.seterr(over='ignore', under='ignore')
        for i in range(len(k_g)):
            out += (delta_values[i] *
                    1 / (1 + self.exp_fun(self.x - thresholds[i], k_g[i])))
        np.seterr(**err_backup)

        return out

    def derivative(self, point):
        """Derivative of this operator."""
        thresholds = point[0:self.num_thresholds]
        values = point[self.num_thresholds:]

        # Create the differences between different threshold values
        delta_values = [values[0] - self.base_value]  # First value is fixed
        delta_values += [values[i] - values[i-1] for i in
                         range(1, self.num_values)]

        # Generate the k_g values
        k_g = [self.sharpness[i] / delta_values[i] for i in
               range(self.num_values)]

        # Create the operator.
        # Create the first outside the loop, in order to be able
        err_backup = np.seterr(over='ignore', under='ignore')
        tmp = self.exp_fun(np.abs(self.x - thresholds[0]), k_g[0])
        ops = ((-2.0 * self.sharpness[0] *
                tmp /
                (1 + tmp)**2) *
               proj(self.domain, 0))

        for i in range(1, self.num_thresholds):
            tmp = self.exp_fun(np.abs(self.x - thresholds[i]), k_g[i])
            ops += ((-2.0 * self.sharpness[i] * tmp / (1 + tmp)**2) *
                    proj(self.domain, i))

        for i in range(self.num_values):
            tmp = self.exp_fun(self.x - thresholds[i], k_g[i])
            tmpABS = self.exp_fun(np.abs(self.x - thresholds[i]), k_g[i])
            ops += ((1 / (1 + tmp) - 2 * self.sharpness[i] *
                     (self.x - thresholds[i]) * tmpABS /
                     ((1 + tmpABS)**2 * delta_values[i])) *
                    proj(self.domain, self.num_thresholds + i))

        np.seterr(**err_backup)

        return ops


class proj(Operator):

    """Helper-cass in order to do coordinate projection `rn` -> `RealNumbers`.
    """

    def __init__(self, space, index):
        """ Initialize and instance."""
        self.index = int(index)
        Operator.__init__(self, space, space.field, True)

    def _call(self, x):
        """Apply the operator."""
        return x[self.index]

    @property
    def adjoint(self):
        """The adjoint operator."""
        orig = self

        class projadj(Operator):

            """Adjoint operator for the helper-class."""

            def _call(self, x):
                """Apply the operator"""
                out = orig.domain.zero()
                out[orig.index] = x
                return out

            @property
            def adjoint(self):
                """The adjoint operator."""
                return orig

        return projadj(self.range, self.domain, True)


class ThresholdOperator(Operator):

    """The thresholding operator.

    A naive implementation of a threshold operator.

    Notes
    -----
    Given a set of values :math:`(a_i)_{i = 1}^{n}` and a set of thresholds
    :math:`(b_i)_{i = 1}^{n-1}` the threshold operator :math:`A : X \\to X` is
    given by

    .. math::
        A(x) = a_i, \\quad \\text{if} \; b_{i-1} \\leq x < b_i,

    where we interpret :math:`b_{0} = -\\infty` and :math:`b_{n} = \\infty`.
    """

    def __init__(self, domain, thresholds, values):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `DiscreteLp` or `FnBase`
            Domain of the operator.
        thresholds : array of float
            The threshold values.
        values : array of float
            The values of operator.

        Examples
        --------
        >>> op = ThresholdOperator(odl.rn(4), [0.5], [0,1])
        >>> op([-1, 0.3, 0.6, 1.2])
        rn(4).element([0.0, 0.0, 1.0, 1.0])
        """
        if not len(thresholds)+1 == len(values):
            raise ValueError('`thresholds` {} needs to be one shorter than '
                             '`values` {}'.format(thresholds, values))

        self.thresholds = thresholds
        self.values = values

        super().__init__(domain=domain, range=domain)

    def thresholding(self, x):
        """Helper function that performs the thresholding."""
        for i in range(len(self.thresholds)):
            if x < self.thresholds[i]:
                return self.values[i]
        return self.values[-1]

    # TODO: This will not work for PorductSpaces
    def _call(self, x):
        """Threshold the values of ``x``."""
        return [self.thresholding(tmp) for tmp in x]


# TODO: this is implemented only for uniform_discrsd
class EdgeDetectOperator(Operator):

    """A naive implementation of an edge detection operator."""

    def __init__(self, domain, diagonal_neighbour=False):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace` or `Field`, optional
            Set of elements on which the operator can be applied.
        """
        self.diagonal_neighbour = diagonal_neighbour

        if not isinstance(domain, DiscreteLp):
            raise NotImplementedError('Onlt works for `uniform_discr`')

        super().__init__(domain=domain, range=domain)

        self.shape_param = domain.shape
        self.elem_len = np.prod(self.shape_param)

        self.forward_grad = Gradient(domain=domain, method='forward',
                                     pad_mode='order0')
        self.backward_grad = Gradient(domain=domain, method='backward',
                                      pad_mode='order0')

    def _call(self, x):
        """Apply the operator."""
        if not self.diagonal_neighbour:
            tmp = (self.forward_grad(x).ufuncs.absolute() +
                   self.backward_grad(x).ufuncs.absolute())

        else:
            raise NotImplementedError('Not done yet')

        return np.greater(sum(tmp), 0)
