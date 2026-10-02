"""specmc: Spectral Mantle Convection.

The package provides

    specmc.ConvectionConfig   the public interface (physical scenario +
                              scenario-dependent numerical parameters)
    specmc.api                config -> validated model, march helpers
    specmc._numerics          the fixed internal numerical constants
    specmc.core               spectral discretisation, CFL policy, implicit
                              theta step, initial conditions
    specmc.physics            viscosity laws (heating / compressibility follow)
    specmc.stokes             eta(z) / eta(x,z) Stokes operators + preconditioners
    specmc.models             the assembled models (make_convection, RBCCoupled)
    specmc.io                 run-tree I/O (atomic checkpoints)

Everything in the package imports only intra-package.  The layout is asserted
by the package self-tests, which also check that no code outside the package
still uses the pre-migration flat module names.

Frozen behaviour
----------------
``specmc.ConvectionConfig()`` reproduces the frozen reference configuration, and the
package self-tests check it key by key against the production driver defaults
and the recorded reference values.  Re-running them re-checks the whole freeze.
"""
from __future__ import annotations

import importlib

from .config import ConvectionConfig
from . import _numerics

__all__ = ["ConvectionConfig", "api", "_numerics"]
__version__ = "0.2.0"


def __getattr__(name):
    """Import ``specmc.api`` on first attribute access (PEP 562).

    Imported lazily: ``api`` pulls in the whole numeric stack (numpy, scipy,
    the solver modules), while ``ConvectionConfig``, ``_numerics`` and ``io``
    are all cheap.  Code that only wants to
    describe a scenario or read a checkpoint should not pay for the solver.

    ``importlib.import_module`` is used rather than ``from . import api``: the
    ``from`` form probes ``hasattr(self, 'api')`` first, which re-enters this
    hook and recurses.
    """
    if name == "api":
        return importlib.import_module(".api", __name__)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
