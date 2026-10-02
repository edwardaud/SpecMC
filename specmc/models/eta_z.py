"""The depth-dependent scenario: ``viscosity_model='eta_z'``.

eta(z) = exp(-ln(deta_T) (1 - z)), depth only, so the Stokes operator is time
independent and the coupled ``eta(T(x,z,t))`` operator is never used
(a structural self-test asserts exactly that).  At deta_T = 1 this branch is
bit-identical to the isoviscous baseline, which is why it is the production
time integrator's constant-physics route.

One of the three explicit recipe modules that :func:`specmc.api.model_spec`
dispatches to; see :mod:`specmc.models.const` for the convention.

The builder stays ``convection_modes.make_convection`` and every numerical
switch comes from :mod:`specmc._numerics`, the values are the literals that
used to live in ``specmc.api.model_spec``, so this module adds no physics.
"""
from __future__ import annotations

from .. import _numerics as NUM
from ..core.implicit_theta import RBCImplicitTheta
from .convection_modes import make_convection

__all__ = ["NAME", "CLS", "BUILDER", "kwargs", "build", "spec"]

#: the ``ConvectionConfig.viscosity_model`` value this module implements
NAME = "eta_z"
#: the class ``specmc.api.build_model`` must get back
CLS = RBCImplicitTheta
#: the factory that constructs the model
BUILDER = "make_convection"


def kwargs(cfg):
    """-> this scenario's constructor keyword arguments (a fresh dict)."""
    return dict(Ra=cfg.Ra, Nx=int(cfg.Nx), Nz=int(cfg.Nz), Lx=cfg.Lx,
                viscosity_law="exp_T_depth", deta_T=float(cfg.deta_T),
                symmetrize=bool(cfg.symmetrize), zform=NUM.ZFORM,
                theta_scheme=NUM.THETA_SCHEME,
                implicit_buoy=NUM.IMPLICIT_BUOY,
                theta_rtol=NUM.THETA_RTOL,
                theta_extrapolate=NUM.THETA_EXTRAPOLATE,
                internal_heating=float(cfg.internal_heating),
                # viscous dissipation: `(Di/Ra) Phi`; Di = 0 (the default) is the frozen reference
                # temperature equation, bit for bit.
                dissipation_number=float(cfg.dissipation_number),
                # TALA: 'boussinesq' (the frozen default) or 'tala'.
                compressibility=str(cfg.compressibility),
                eta_min=float(cfg.eta_min), eta_max=float(cfg.eta_max),
                eta_smooth=float(cfg.eta_smooth))


def build(**kw):
    """-> the model instance."""
    return make_convection(**kw)


def spec(cfg, compressibility):
    """-> the full recipe dict consumed by :func:`specmc.api.model_spec`."""
    return dict(model=NAME, compressibility=compressibility, builder=BUILDER,
                cls=CLS, kwargs=kwargs(cfg))
