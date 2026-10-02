"""Internal numerical constants for `specmc`.  Not part of the public interface.

Everything in this module is a *fixed implementation decision*.  The production
route has exactly one validated choice for each of these.  Exposing them would
only create a way to configure the solver into a state that does not match
the frozen reference configuration.

The public counterpart is `specmc/config.py` (``ConvectionConfig``).

Provenance
----------
Every value below is the value used by the frozen reference production run, and the
package self-tests re-derive each of them from three independent places

  1. the production driver, statically (argparse defaults + ``make_model()``
     keyword arguments),
  2. the ``config`` dict that the reference run recorded about itself,
  3. the frozen configuration table,

so a drift in any one of them is caught.

Single source of truth
----------------------
The CFL policy already has an SSOT, ``specmc/core/cfl_policy.py``; the two
constants that live there are *imported* here (never re-typed).  ``DTMAX`` and
``DTMIN`` are
also the defaults of ``cfl_policy.cfl_dt``, and the internal cross-checks
compare them against ``cfl_policy.cfl_dt.__defaults__``.

Legacy name mapping
-------------------
    legacy            here
    _STOKES_MODE      STOKES_MODE
    _THETA_SCHEME     THETA_TIME_SCHEME  (the legacy 'bdf2' is the
                                         ImplicitThetaSolver time scheme; the
                                         model-level switch is
                                         THETA_SCHEME = 'implicit_bdf2')
    _IMPLICIT_BUOY    IMPLICIT_BUOY
    _CFL_SAFETY       CFL_SAFETY        (= cfl_policy.CFL_SAFETY_DEFAULT)
    _CFL_MAX          CFL_MAX           (= cfl_policy.CFL_SAFETY_MAX)
    _ETA_DEALIAS      ETA_DEALIAS
    _DTMAX            DTMAX
"""
from __future__ import annotations

#: imported intra-package: the SSOT is specmc/core/cfl_policy.py
from .core.cfl_policy import (CFL_SAFETY_DEFAULT as _CFL_SAFETY_DEFAULT,
                              CFL_SAFETY_MAX as _CFL_SAFETY_MAX)

__all__ = [
    "STOKES_MODE", "PARTNER_SIGN", "ZFORM", "ZSOLVER", "USE_LU", "DEALIAS",
    "DEALIAS_Z", "STOKES_RTOL", "ETA_DEALIAS", "ETA_DRIFT_TOL", "ETA_T_CLIP",
    "THETA_SCHEME", "THETA_TIME_SCHEME", "THETA_EXTRAPOLATE", "IMPLICIT_BUOY",
    "THETA_RTOL", "THETA_MAXITER", "THETA_RESTART",
    "CFL_SAFETY", "CFL_MAX", "CFL_FALLBACK", "DTMAX", "DTMIN",
    "TSTOP", "AMP_LIMIT", "STEADY_TOL", "STEADY_WINDOW", "CKPT_DT",
    "CKPT_KEEP_SPACING", "DEFECT_ADAPTIVE",
    "as_dict",
]

# --------------------------------------------------------------------------
# 1. Stokes / coupled-operator route (the corrected production operator)
# --------------------------------------------------------------------------
#: 'defect' = one lagged factorisation + adaptive Picard defect correction.
#: 'direct' = assemble + dense LU at every step; a *reference* route that only
#: the internal cross-check scripts select (never the public interface).
STOKES_MODE = "defect"
#: x-parity of the Stokes stream function: conj(psi_m) = partner_sign * psi_m.
#: -1 is the physical (x-odd) class; 0 is the earlier shipped operator.  This is
#: a property of the field, not a tunable.
PARTNER_SIGN = -1.0
#: 'reduced' = block-eliminated 2nd-order operator (conditioning ~N^4);
#: 'tau4' = the 4th-order reference operator.
ZFORM = "reduced"
#: Chebyshev-tau z discretisation (the 'galerkin' alternative is a reference
#: implementation of the same discrete operator).
ZSOLVER = "tau"
#: LU caching of the per-wavenumber solvers (pure performance; does not change
#: the physics).
USE_LU = True
#: x-direction 3/2 dealiasing of the quadratic nonlinearity.
DEALIAS = True
#: z-direction zero-padded dealiasing.  OFF in production; it is mutually
#: exclusive with theta_scheme='implicit_bdf2' (RBCImplicitTheta raises).
DEALIAS_Z = False
#: relative tolerance of the matrix-free FGMRES inner solve of the coupled
#: Stokes operator (LaggedStokes.rtol).
STOKES_RTOL = 1e-8
#: eta = exp(-ln(deta_T) T) is a nonlinear function of theta, so it is
#: synthesised on ETA_DEALIAS * Nx points and truncated back to Nk modes
#: (coupled_model.eta_of).  1 is the old, aliased pointwise path.
ETA_DEALIAS = 2
#: Pointwise relative eta drift that forces a fresh factorisation inside the
#: lag window (production value 0.1; the class default 0.05 is NOT the reference).
ETA_DRIFT_TOL = 0.1
#: Temperature clip of the CURRENT implementation (coupled_model._eta_from_T).
#: This is what bounds eta today; it is unrelated to the
#: separate `eta_min` / `eta_max` window, which is only recorded (see config.py).
ETA_T_CLIP = (0.0, 2.0)

# --------------------------------------------------------------------------
# 2. Temperature time integration (BDF2 + 2nd-order velocity extrapolation)
# --------------------------------------------------------------------------
#: model-level switch passed to RBCImplicitTheta (reference production value).
THETA_SCHEME = "implicit_bdf2"
#: time scheme of the inner ImplicitThetaSolver that THETA_SCHEME maps to
#: (the legacy `_THETA_SCHEME = 'bdf2'`).
THETA_TIME_SCHEME = "bdf2"
#: second-order extrapolation of the advection velocity and the buoyancy
#: source; without it BDF2 degrades to first order.
THETA_EXTRAPOLATE = True
#: reference production moves NO part of the buoyancy to the implicit side.
IMPLICIT_BUOY = False
#: FGMRES tolerance of the implicit theta solve (never overridden by any
#: driver of the project).
THETA_RTOL = 1e-10
THETA_MAXITER = 200
THETA_RESTART = 40

# --------------------------------------------------------------------------
# 3. CFL / dt policy (SSOT = cfl_policy.py; imported, never re-typed)
# --------------------------------------------------------------------------
#: FAC used on the first attempt: dt = min(DTMAX, CFL_SAFETY * global CFL).
CFL_SAFETY = _CFL_SAFETY_DEFAULT                      # 40.0
#: hard cap of the safety factor (clamp_safety).
CFL_MAX = _CFL_SAFETY_MAX                             # 80.0
#: FAC used after an instability (driver --fac-fb = 0.5 * CFL_SAFETY_DEFAULT).
CFL_FALLBACK = 0.5 * _CFL_SAFETY_DEFAULT              # 20.0
#: dt ceiling of the production policy (raised from an earlier 1e-5).
DTMAX = 4.0e-5
#: dt floor of the production policy.
DTMIN = 1e-10

# --------------------------------------------------------------------------
# 4. Run control of the frozen reference march (not physics; reference driver values)
# --------------------------------------------------------------------------
#: physical end time of the 64x96 case-2a benchmark.
TSTOP = 0.25
#: abort when max|theta| exceeds this multiple of the initial amplitude.
AMP_LIMIT = 5.0
#: steady-stop criterion |Nu_energy - Nu_top| < STEADY_TOL for STEADY_WINDOW.
STEADY_TOL = 1e-3
STEADY_WINDOW = 0.02
#: checkpoint cadence and the spacing of the kept (resume) checkpoints.
CKPT_DT = 0.01
CKPT_KEEP_SPACING = 0.05
#: adaptive (defect_tol) vs fixed-K defect correction; the reference uses adaptive.
DEFECT_ADAPTIVE = True


def as_dict():
    """-> every internal constant as a plain ``{name: value}`` dict.

    Used by ``specmc.api.numerics()`` (logging) and by the package self-tests,
    which compare the table against the production driver defaults and the
    recorded reference values.
    """
    out = {}
    for name, value in globals().items():
        if name.startswith("_") or not name.isupper():
            continue
        if isinstance(value, (bool, int, float, str, tuple, list, type(None))):
            out[name] = value
    return out
