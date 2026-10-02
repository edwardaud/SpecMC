"""The isoviscous scenario: ``viscosity_model='const'``.

One of the three explicit recipe modules that :func:`specmc.api.model_spec`
dispatches to (the others are :mod:`specmc.models.eta_z` and
:mod:`specmc.models.eta_T`).  Each module owns exactly one translation

    ConvectionConfig  ->  (builder, class, constructor kwargs)

so that "physical scenario -> solver arguments" is one self-contained file per
scenario instead of one branch of a long ``if`` chain in `specmc.api`.

This branch is the frozen isoviscous regression baseline: it must stay the
reference solver ``specmc.core.rbc_galerkin.RBC`` (explicit-implicit diffusion
step, ``RBC.cfl_dt``); an internal check verifies the class identity and the
stored baseline numbers.  Constant physics stepped by the *production* time
integrator is available as ``viscosity_model='eta_z', deta_T=1``, the
"immune region" equivalence of the frozen production configuration.

Nothing here changes physics: the kwargs are the literals that used to live in
``specmc.api.model_spec``.
"""
from __future__ import annotations

from ..core.rbc_galerkin import RBC
from .convection_modes import make_convection

__all__ = ["NAME", "CLS", "BUILDER", "kwargs", "build", "spec"]

#: the ``ConvectionConfig.viscosity_model`` value this module implements
NAME = "const"
#: the class ``specmc.api.build_model`` must get back
CLS = RBC
#: the factory that constructs the model
BUILDER = "make_convection"


def kwargs(cfg):
    """-> this scenario's constructor keyword arguments (a fresh dict)."""
    return dict(Ra=cfg.Ra, Nx=int(cfg.Nx), Nz=int(cfg.Nz), Lx=cfg.Lx,
                viscosity_law="isoviscous", deta_T=float(cfg.deta_T),
                internal_heating=float(cfg.internal_heating),
                # viscous dissipation: `(Di/Ra) Phi` on the isoviscous (eta = 1) solver; the
                # default 0.0 leaves the frozen baseline bit for bit.
                dissipation_number=float(cfg.dissipation_number),
                # TALA: 'boussinesq' (the frozen default) or 'tala'.
                compressibility=str(cfg.compressibility))


def build(**kw):
    """-> the model instance."""
    return make_convection(**kw)


def spec(cfg, compressibility):
    """-> the full recipe dict consumed by :func:`specmc.api.model_spec`."""
    return dict(model=NAME, compressibility=compressibility, builder=BUILDER,
                cls=CLS, kwargs=kwargs(cfg))
