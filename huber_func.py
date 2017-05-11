#

"""Implementation of the Huber norm """

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

from odl.solvers.functional.functional import Functional
from odl.operator import Operator


class HuberNorm(Functional):

    """The Huber functional"""

    def __init__(self, space, epsilon):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `FnBase`
            Domain of the functional.
        epsilon : float
            The parameter of the Huber functional.
        """
        self.__epsilon = float(epsilon)
        super().__init__(space=space, linear=False, grad_lipschitz=2)

    @property
    def epsilon(self):
        """The parameter of the Huber functional."""
        return self.__epsilon

    def _call(self, x):
        """Return the squared Huber norm of ``x``."""
        indices = x.ufuncs.absolute().asarray() < self.epsilon
        indices = np.float32(indices)

        tmp = ((x * indices)**2 / (2.0 * self.epsilon) +
               (x.ufuncs.absolute() - self.epsilon / 2.0) * (1-indices))

        return tmp.inner(self.domain.one())

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        functional = self

        class HuberNormGradient(Operator):

            """The gradient operator of this functional."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

            # TODO: Update this call. Might not work for PorductSpaces
            def _call(self, x):
                """Apply the gradient operator to the given point."""
                indices = x.ufuncs.absolute().asarray() < functional.epsilon
                indices = np.float32(indices)

                tmp = ((x * indices) / (functional.epsilon) +
                       (x).ufuncs.sign() * (1-indices))

                return tmp

        return HuberNormGradient()

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.domain)
