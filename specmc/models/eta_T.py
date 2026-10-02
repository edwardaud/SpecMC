"""The temperature-dependent scenario: ``viscosity_model='eta_T'``.

eta = exp(-ln(deta_T) T(x,z,t)), the full coupled problem: the corrected
coupled Stokes operator (``specmc.stokes.stokes_coupled``), the lagged
factorisation with the adaptive Picard defect correction
(``specmc.stokes.lagged_stokes``) and the BDF2 implicit temperature step
(``specmc.core.implicit_theta``).

This is the frozen reference production configuration and the default of
``ConvectionConfig``; an internal check proves key by key that the solver
defaults match the frozen production state.

The Arrhenius law: the SAME module also serves ``viscosity_model='arrhenius'``.  That
scenario differs from ``'eta_T'`` in exactly one place, the law evaluated in
``RBCCoupled._eta_from_T``, and its `kwargs` therefore only add
``arrhenius_Ts`` / ``arrhenius_E`` (both 0.0 by default, which selects the
frozen exponential branch bit for bit).  Everything else, including the
numerics switches below, is shared on purpose: the two laws are calibrated to
the SAME endpoints, so the comparison between them is a single-variable
experiment.

One of the explicit recipe modules that :func:`specmc.api.model_spec`
dispatches to; see :mod:`specmc.models.const` for the convention.  Every
numerical switch comes from :mod:`specmc._numerics` (or from the
``ConvectionConfig`` for the scenario-dependent ones), so this module adds no
physics: the kwargs are the literals that used to live in
``specmc.api.model_spec``.
"""
from __future__ import annotations

from .. import _numerics as NUM
from .coupled_model import RBCCoupled

__all__ = ["NAME", "CLS", "BUILDER", "kwargs", "build", "spec"]

#: the ``ConvectionConfig.viscosity_model`` value this module implements
NAME = "eta_T"
#: the class ``specmc.api.build_model`` must get back
CLS = RBCCoupled
#: the constructor that builds the model (no factory indirection)
BUILDER = "RBCCoupled"


def kwargs(cfg):
    """-> this scenario's constructor keyword arguments (a fresh dict)."""
    dtol = float(cfg.defect_tol)
    return dict(Ra=cfg.Ra, Nx=int(cfg.Nx), Nz=int(cfg.Nz), Lx=cfg.Lx,
                deta_T=float(cfg.deta_T),
                symmetrize=bool(cfg.symmetrize), zform=NUM.ZFORM,
                theta_scheme=NUM.THETA_SCHEME,
                theta_extrapolate=NUM.THETA_EXTRAPOLATE,
                implicit_buoy=NUM.IMPLICIT_BUOY,
                stokes_mode=NUM.STOKES_MODE,
                lag=int(cfg.lag), K=int(cfg.K),
                stokes_rtol=NUM.STOKES_RTOL,
                defect_tol=(None if dtol <= 0.0 else dtol),
                max_K=int(cfg.max_K),
                eta_drift_tol=NUM.ETA_DRIFT_TOL,
                eta_dealias=NUM.ETA_DEALIAS,
                internal_heating=float(cfg.internal_heating),
                # viscous dissipation: the viscous-dissipation number Di of
                # `specmc.physics.dissipation`; the source is `(Di/Ra) Phi`.
                # Default 0.0, and `Di = 0` never evaluates it, so the frozen
                # reference path stays bit for bit.
                dissipation_number=float(cfg.dissipation_number),
                # TALA: 'boussinesq' (the frozen default) or 'tala'.  The TALA
                # reference state is built from the SAME `dissipation_number`,
                # so no new physics parameter appears here.
                compressibility=str(cfg.compressibility),
                # the Arrhenius (T-only) law.  Both default to 0.0, and
                # `arrhenius_Ts = 0` makes the model take the frozen reference
                # exponential branch (bit for bit).  This recipe module is
                # shared by `viscosity_model='eta_T'` and `'arrhenius'`, which
                # is exactly the point: the two scenarios differ ONLY in how
                # eta(T) is evaluated, not in the numerics.
                arrhenius_Ts=float(cfg.arrhenius_Ts),
                arrhenius_E=float(cfg.arrhenius_E),
                # the pressure-dependent viscosity law: the reference-hydrostatic pV(L) term.  Default 0.0, and
                # `arrhenius_V = 0` never enters the pV branch, so both the
                # the Arrhenius T-only path and the frozen reference path stay bit for bit.
                # The dimensional constants come from the frozen
                # `reference_state.REFERENCE_DEFAULTS` contract (not from the
                # config: they are a physics contract, not a public knob).
                arrhenius_V=float(cfg.arrhenius_V),
                # the pressure-dependent viscosity law: which third condition the pV calibration uses.
                # 'D' (default) keeps the frozen contrast; 'Ea' makes D the
                # output.  Irrelevant when arrhenius_V == 0.
                reference_anchor=str(cfg.reference_anchor),
                eta_min=float(cfg.eta_min), eta_max=float(cfg.eta_max),
                eta_smooth=float(cfg.eta_smooth),
                precond=str(cfg.precond).strip().lower())


def build(**kw):
    """-> the model instance."""
    return RBCCoupled(**kw)


def spec(cfg, compressibility):
    """-> the full recipe dict consumed by :func:`specmc.api.model_spec`."""
    return dict(model=NAME, compressibility=compressibility, builder=BUILDER,
                cls=CLS, kwargs=kwargs(cfg))
