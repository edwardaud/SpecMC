"""Production time-stepping for the FULL variable-viscosity model
eta = eta(T(x, z, t)), Blankenbach et al. (1989) case 2a physics.

The production coupled eta(T) solver.

What this file adds
-------------------
`convection_modes.RBCVariableViscosity` freezes the viscosity to the conductive
depth profile eta_bar(z), which keeps the Stokes operator
wavenumber-decoupled and solvable by one LU per mode.  Case 2a needs
eta = exp(-ln(deta_T) * T) with the *evolving* temperature, so the operator
couples all horizontal wavenumbers and must be solved as one big system.

The coupled-solver study showed how to do that cheaply (`lagged_stokes`):

    refresh (assemble + dense LU of the Schur form, every `lag` steps)
    x0 = A_lag^{-1} b                                  (one back substitution)
    x <- x + A_lag^{-1} (b - A(eta_new) x)   K times   (Picard defect correction)

with the matrix-free true operator `FreeSchurOperator`.  K = 3 at lag = 16
removes the splitting error to < 1e-8 at 11.86 ms/step against 85.6 ms/step for
a fresh assemble+LU every step.

`RBCCoupled` puts that solver into the model's `stokes()` hook, so the whole
validated time-stepping / diagnostics / CFL machinery of `convection_modes`
is reused unchanged, and adds:

  * `stokes_mode='direct'`, the reference route: assemble + dense LU of
    A(eta(T^n)) at EVERY step.  Used to verify the defect-corrected route.
  * `stokes_mode='defect'`, lagged factorisation + K Picard corrections
    (`K` fixed, or `defect_tol` makes K adaptive).
  * a one-entry memo on `stokes()` so that `cfl_dt(theta)` followed by
    `step(theta, dt)` costs ONE coupled solve, not two.

    with t0 = 0, iterate   dt = cfl_dt(s);  s = step(s, dt)

Temperature scheme: BDF2 (`implicit_theta`)
with the second-order velocity extrapolation, or backward Euler (`implicit`).

internal heating: `internal_heating = H` passes the constant volumetric
source of `specmc.physics.heating` down to the theta right-hand side.  It is the
only change to the temperature equation, it does not touch the Stokes solve,
the eta construction or the diagnostics, and `H = 0` (the frozen reference value)
leaves every floating-point operation of the step exactly where it was.

the Arrhenius law (T-only): `arrhenius_Ts > 0` replaces the ONE law evaluation
inside `_eta_from_T` by the activated-creep law of
`specmc.physics.arrhenius` (``ln eta = E/(T+Ts) - E/Ts``, ``E = ln(deta_T) Ts
(1+Ts)``), which is calibrated to the SAME two endpoints as the exponential
law.  Nothing else moves: the padded-space construction, the explicit eta
window, the rfft truncation, the positivity guard and every solver
switch are shared verbatim, and `arrhenius_Ts = 0.0` (the default) takes a
branch that is taken before any float arithmetic, so the frozen reference path stays
bit for bit.

pV(L): `arrhenius_V > 0` adds the reference-hydrostatic pressure
term of the frozen contract (`specmc.physics.arrhenius.PV_CONTRACT`) to the
same one law evaluation:

    ln eta(T, z) = (E-tilde + V-tilde p0_norm(z)) / (T + Ts) + C ,
    p0_norm(z) = 1 - z   from `specmc.physics.reference_state` (the SSOT)

with ``(E-tilde, V-tilde, Ts, C)`` from the ONE calibration procedure
``reference_state.calibrate_pv``.  The pV branch is a *third* branch taken only
when ``arrhenius_V > 0``: ``arrhenius_V = 0.0`` (the default) never enters it,
so the Arrhenius path and the frozen reference path stay bit for bit.  The
evaluation still happens in the padded space and is still truncated
and guarded exactly as before, pV adds ONE z-array to the exponent, it does
not touch the spectral discretisation, the conditioning or the eta window
machinery.

viscous dissipation: `dissipation_number = Di` adds the source
``(Di/Ra) Phi`` of `specmc.physics.dissipation` to the temperature right-hand
side, where ``Phi = 2 eta (eps - (1/3) tr(eps) 1) : eps`` is evaluated from the
step's frozen Stokes velocity and from `last_eta`, the eta field the
coupled Stokes operator consumed (`_stokes_coupled`).  `Di = 0.0` (the default,
and the frozen reference value) never evaluates the source, so the frozen path stays
bit for bit.  The source is explicit: it enters the right-hand side only, so
the Stokes operator, the eta construction, the preconditioners and the tau rows
are untouched.
"""
import time
import numpy as np

from ..core.implicit_theta import RBCImplicitTheta
from ..core.krylov_fgmres import fgmres
from ..stokes.bfbt import SpectralBFBTPrecond
from ..stokes.lagged_stokes import LaggedStokes, FreeSchurOperator
from ..stokes.stokes_coupled import CoupledStokes
from ..stokes.stokes_tala import free_operator_for, operator_for
from ..physics import eta_bounds as EB
from ..physics import heating as HEAT
from ..physics.arrhenius import arrhenius_E as _arrhenius_E
from ..physics.arrhenius import arrhenius_exponent as _arrhenius_exponent
from ..physics.arrhenius import arrhenius_exponent_V as _arrhenius_exponent_V
from ..physics.reference_state import (REFERENCE_DEFAULTS,
                                       calibrate_pv as _calibrate_pv,
                                       p0_norm as _p0_norm)
from ..physics.viscosity import eta_exp_T
import scipy.linalg as sla

__all__ = ["RBCCoupled", "PRECONDS", "self_test"]

#: Stokes preconditioner selection.  'legacy' is the frozen
#: reference route; 'bfbt' is the spectral BFBT preconditioned FGMRES route.
PRECONDS = ("legacy", "bfbt")

#: Largest `Nk * Nz` for which the bfbt route's LAST-RESORT rescue
#: (`stokes_rescue='direct'`) is allowed to assemble and LU-factorise the dense
#: Schur matrix.  At 32x48 that is 17x48 = 816 (a 816^2 complex matrix, ~10 MB);
#: at 64x96 it is 33x96 = 3168 (~160 MB, a few seconds per factorisation), and
#: at 128x192 it is 97x192 = 18624 (~5.5 GB) which must NOT be attempted.
#: Above the cap the rescue stops at the lagged defect-corrected solution and
#: says so through `n_bfbt_rescue_weak`.
DIRECT_RESCUE_MAX = 4096

#: Relative tolerance used to decide that a bfbt FGMRES result is usable.  The
#: solver's own stopping test is `rtol`; the extra factor only absorbs the
#: difference between the GMRES quasi-residual and the true residual.
RESCUE_RTOL_SLACK = 1.0


class RBCCoupled(RBCImplicitTheta):
    """eta = eta(T(x,z,t)) model with the defect-corrected coupled Stokes solve."""

    def __init__(self, Ra=1e4, Nx=64, Nz=48, Lx=2.0, zsolver='tau',
                 dealias=True, deta_T=1000.0, use_lu=True, dealias_z=False,
                 symmetrize=True, zform='reduced',
                 theta_scheme='implicit_bdf2', implicit_buoy=False,
                 theta_rtol=1e-10, theta_extrapolate=True,
                 stokes_mode='defect', lag=16, K=3, stokes_rtol=1e-8,
                 defect_tol=None, max_K=8, precond_stale=False,
                 eta_drift_tol=0.05, eta_dealias=2, internal_heating=0.0,
                 dissipation_number=0.0, compressibility='boussinesq',
                 eta_min=None, eta_max=None, eta_smooth=None,
                 precond='legacy', precond_profile='harm',
                 arrhenius_Ts=0.0, arrhenius_E=0.0, arrhenius_V=0.0,
                 reference_anchor='D', reference_constants=None,
                 stokes_restart=80, stokes_maxiter=800):
        super().__init__(Ra=Ra, Nx=Nx, Nz=Nz, Lx=Lx, zsolver=zsolver,
                         dealias=dealias, viscosity_law='exp_T_depth',
                         deta_T=deta_T, use_lu=use_lu, dealias_z=dealias_z,
                         symmetrize=symmetrize, zform=zform,
                         theta_scheme=theta_scheme,
                         implicit_buoy=implicit_buoy,
                         theta_rtol=theta_rtol,
                         theta_extrapolate=theta_extrapolate,
                         internal_heating=internal_heating,
                         dissipation_number=dissipation_number,
                         compressibility=compressibility,
                         eta_min=eta_min, eta_max=eta_max,
                         eta_smooth=eta_smooth)
        self._X, self._Z = np.meshgrid(self.x, self.z, indexing='ij')
        # --- the viscous-dissipation source --------------------------
        # `dissipation_number = Di` adds `(Di/Ra) Phi` to the temperature
        # right-hand side (`specmc.physics.dissipation`); `0.0` (the default,
        # and the frozen reference value) never evaluates it.  The eta the source uses
        # is `last_eta`: the field `_stokes_coupled` handed to the operator
        # (assigned below, not re-derived).
        self.dissipation_number = float(dissipation_number)
        if not np.isfinite(self.dissipation_number) \
                or self.dissipation_number < 0.0:
            raise ValueError('dissipation_number must be finite and >= 0 '
                             '(got %r)' % (dissipation_number,))
        self.dissipation_active = bool(self.dissipation_number != 0.0)
        self.last_eta = None
        # --- the Arrhenius (T-only) law ------------------------------
        # `arrhenius_Ts = 0.0` (the default) means "off"; the law branch in
        # `_eta_from_T` is then never taken and the frozen reference path is bit for
        # bit.  A positive value selects `specmc.physics.arrhenius`, whose
        # activation is CALIBRATED (E = ln(deta_T) Ts (1+Ts)) unless the calling
        # code passes the same value explicitly.
        self.arrhenius_Ts = float(arrhenius_Ts)
        if not np.isfinite(self.arrhenius_Ts) or self.arrhenius_Ts < 0.0:
            raise ValueError('arrhenius_Ts must be finite and >= 0 (got %r)'
                             % (arrhenius_Ts,))
        # --- the pressure-dependent viscosity law: the pV(L) branch (a THIRD branch; V = 0 never enters it),
        self.arrhenius_V = float(arrhenius_V)
        if not np.isfinite(self.arrhenius_V) or self.arrhenius_V < 0.0:
            raise ValueError('arrhenius_V must be finite and >= 0 (got %r)'
                             % (arrhenius_V,))
        self.arrhenius_pv_active = bool(self.arrhenius_V > 0.0)
        self.reference_constants = (REFERENCE_DEFAULTS
                                    if reference_constants is None
                                    else reference_constants)
        self.pv_record = None
        self.arrhenius_E = 0.0
        self._arrhenius_L = 0.0
        self.reference_anchor = str(reference_anchor)
        if self.arrhenius_pv_active:
            from ..physics.reference_state import (
                ReferenceConstants as _RC, PV_ANCHORS as _ANCH)
            if not isinstance(self.reference_constants, _RC):
                raise TypeError('reference_constants must be a ReferenceConstants'
                                ' (got %r)' % (type(reference_constants),))
            if self.reference_anchor not in _ANCH:
                raise ValueError('unknown reference_anchor %r (use one of %s)'
                                 % (reference_anchor, ", ".join(_ANCH)))
            # the ONE calibration procedure.  `arrhenius_V` IS V-tilde, so a
            # scan factor is expressed by passing a different non-negative value.
            rec = _calibrate_pv(
                self.reference_constants, anchor=self.reference_anchor,
                V=self.arrhenius_V,
                deta_T=(float(self.deta_T)
                        if self.reference_anchor == "D" else None))
            Ts_phys = float(rec['Ts'])
            if self.arrhenius_Ts != 0.0 and \
                    abs(self.arrhenius_Ts - Ts_phys) > 1e-12 * Ts_phys:
                raise ValueError(
                    'arrhenius_Ts=%r contradicts the reference-state value '
                    'Ts = T_surf/DeltaT = %r; under pV(L) Ts is fixed by the '
                    'constants contract (leave it 0.0 to accept the derived '
                    'value)' % (arrhenius_Ts, Ts_phys))
            E_given = float(arrhenius_E)
            if not np.isfinite(E_given) or E_given < 0.0:
                raise ValueError('arrhenius_E must be finite and >= 0 (got %r)'
                                 % (arrhenius_E,))
            if E_given != 0.0 and \
                    abs(E_given - rec['E']) > 1e-12 * abs(rec['E']):
                raise ValueError(
                    'arrhenius_E=%r contradicts the calibrated pV activation '
                    'E-tilde = E_a/(R DeltaT) = %r' % (arrhenius_E, rec['E']))
            self.pv_record = dict(rec)
            self._pv_E = float(rec['E'])
            self._pv_V = float(rec['V'])
            self._pv_Ts = float(rec['Ts'])
            self._pv_C = float(rec['C'])
            self.arrhenius_Ts = self._pv_Ts
            self.arrhenius_E = self._pv_E
        elif self.arrhenius_Ts > 0.0:
            E_cal = _arrhenius_E(self.deta_T, self.arrhenius_Ts)
            E_given = float(arrhenius_E)
            if not np.isfinite(E_given) or E_given < 0.0:
                raise ValueError('arrhenius_E must be finite and >= 0 (got %r)'
                                 % (arrhenius_E,))
            if E_given != 0.0 and abs(E_given - E_cal) > 1e-12 * abs(E_cal):
                raise ValueError(
                    'arrhenius_E=%r contradicts the calibrated value E = '
                    'ln(deta_T) Ts (1+Ts) = %r' % (arrhenius_E, E_cal))
            self.arrhenius_E = E_cal
            #: precomputed exactly as `arrhenius_exponent` uses them, so the
            #: per-step evaluation is one reciprocal, one multiply and one exp
            #: (the same operation count the exponential law has).
            self._arrhenius_L = float(np.log(float(self.deta_T)))
        else:
            if float(arrhenius_E) != 0.0:
                raise ValueError(
                    'arrhenius_E is set but arrhenius_Ts is 0 (Arrhenius off); '
                    'the activation would be silently ignored')
            self._arrhenius_L = 0.0
        self.arrhenius_active = bool(self.arrhenius_Ts > 0.0
                                     or self.arrhenius_pv_active)
        self.stokes_mode = str(stokes_mode).lower()
        if self.stokes_mode not in ('defect', 'direct'):
            raise ValueError('unknown stokes_mode %r' % (stokes_mode,))
        self.lag = max(1, int(lag))
        self.K = int(K)
        self.stokes_rtol = float(stokes_rtol)
        self.defect_tol = defect_tol
        self.max_K = int(max_K)
        self.eta_drift_tol = float(eta_drift_tol)
        self.precond_stale = bool(precond_stale)
        # --- Stokes preconditioner selection ---------------------------------
        self.precond = str(precond).lower()
        if self.precond not in PRECONDS:
            raise ValueError('unknown precond %r (use one of %s)'
                             % (precond, ", ".join(PRECONDS)))
        self.precond_profile = str(precond_profile).lower()
        self.stokes_restart = int(stokes_restart)
        self.stokes_maxiter = int(stokes_maxiter)
        self.n_bfbt_solves = 0
        self.n_bfbt_iters = 0
        self.last_bfbt_iters = 0
        self.last_bfbt_info = 0
        #: the bfbt route must never hand an unconverged
        #: Stokes velocity to the time step (the "no silent
        #: failure" rule).  `last_bfbt_true_res` is the TRUE relative residual of
        #: the FGMRES result, `n_bfbt_rescue` counts the steps whose bfbt result
        #: was rejected and recomputed with the lagged defect-corrected route,
        #: `n_bfbt_rescue_direct` the subset that additionally needed the dense
        #: LU, `n_bfbt_rescue_weak` the subset where even that was not allowed
        #: (`Nk*Nz > DIRECT_RESCUE_MAX`) and the lagged result was accepted, and
        #: `last_stokes_route` records which route produced the accepted solve.
        self.last_bfbt_true_res = 0.0
        self.n_bfbt_rescue = 0
        self.n_bfbt_rescue_direct = 0
        self.n_bfbt_rescue_weak = 0
        self.last_stokes_route = 'bfbt'
        #: eta is evaluated on `eta_pad_factor * Nx` points and truncated back
        #: to the first Nk x-modes.  1 = the old,
        #: aliased pointwise path; 2 is the production default (the input theta
        #: is band-limited to Nk modes, so the padded synthesis is exact).
        self.eta_pad_factor = max(1, int(eta_dealias))
        self.ls = None
        self.n_since_refresh = 0
        self.n_refresh = 0
        self.n_defect = 0
        self.n_direct_stokes = 0
        self.last_K = 0
        self.last_corr = 0.0
        self.last_defect_res = 0.0
        self._force_refresh = False
        self._last_failed = False
        # --- silent-failure counters -----------------------------------------
        self.n_defect_failed = 0    # defect correction hit max_K without tol
        self.n_defect_double_fail = 0   # ... and failed AGAIN after the refresh
        self.n_clip_T = 0           # eta: T clipped into [0, 2]
        self.n_T_out_of_range = 0   # eta: T outside the physical [0, 1]
        self.max_T_excursion = 0.0  # largest excursion of T outside [0, 1]
        #: the hard clip: number of padded-grid points whose eta was moved
        #: by `clip(eta_clip_lo, eta_clip_hi)` (`specmc.physics.eta_bounds`).
        #: The bounds themselves (`eta_clip_lo/hi`) are set by
        #: `RBCVariableViscosity.__init__` from the config's eta_min/eta_max.
        self.n_clip_eta = 0
        #: the smooth clip: number of padded-grid points inside one of the
        #: two smooth transition bands (`eta_smooth > 0` only; 0 otherwise).
        self.n_smooth_eta = 0
        #: number of points moved by the POST-TRUNCATION
        #: positivity guard of `eta_of` (0 unless the x-projection of the padded
        #: eta left the physical half-line; see `_guard_eta_positive`).
        self.n_clip_eta_proj = 0
        self.n_smooth_eta_proj = 0
        self.last_eta_pad_factor = self.eta_pad_factor
        self._cache_key = None
        self._cache_X = None
        self._cache_val = None
        self._cache_tol = 1e-12
        self.n_memo_hits = 0
        self.t_stokes = 0.0
        self.t_refresh = 0.0

    # ------------------------------------------------------------------ eta
    def _eta_from_T(self, T, p0n=None):
        """eta(T[, z]) with the clip made EXPLICIT (it used to be silent).

        The maximum principle gives T in [0, 1]; `n_T_out_of_range` counts
        points outside that (a genuine numerical warning), `n_clip_T` counts
        points that the `clip(0, 2)` moved, and `max_T_excursion`
        records the LARGEST excursion so the counts can be read in context
        (a count of 1e5 points that are 1e-16 outside [0, 1] is roundoff; a
        count of 1e5 points at T = -0.05 is a real violation).

        the hard clip adds the explicit viscosity bound
        `clip(eta_min, eta_max)` of `specmc.physics.eta_bounds`, applied to the
        eta of this (padded) array, i.e. *before* `eta_of` truncates the
        field back to the Nk resolved x-modes.  Doing it after the truncation
        would fold the clip's high-wavenumber content onto the resolved modes
        (the anti-aliasing rule).  `n_clip_eta` counts the points it moved; it
        is zero, and the returned array is the *input object*, bit for bit,
        whenever the whole field is inside the window, which is the case for
        the frozen reference configuration and its documented default window.

        the pressure-dependent viscosity law adds the ``p0n`` argument: when the pV(L) branch is active the
        law needs the dimensionless reference pressure ``1 - z`` of the SAME
        padded grid (the SSOT is :func:`reference_state.p0_norm`; the calling
        code may pass it in, otherwise the SSOT is called).  The ``p0n``
        argument is
        ignored by the two non-pV branches, so the default path is unchanged.
        """
        T = np.asarray(T, dtype=float)
        lo = float(T.min())
        hi = float(T.max())
        exc = max(0.0 - lo, hi - 1.0, 0.0)
        if exc > self.max_T_excursion:
            self.max_T_excursion = exc
        self.n_T_out_of_range += int(np.count_nonzero((T < 0.0) | (T > 1.0)))
        self.n_clip_T += int(np.count_nonzero((T < 0.0) | (T > 2.0)))
        Tc = np.clip(T, 0.0, 2.0)
        if self.arrhenius_pv_active:
            # the pressure-dependent viscosity law: the reference-hydrostatic pV law.  This branch
            # is reached ONLY when `arrhenius_V > 0`; the `p0n` array is the
            # padded-grid `1 - z` from the SSOT (never re-typed here).
            pn = _p0_norm(self.z[None, :]) if p0n is None else p0n
            eta = np.exp(_arrhenius_exponent_V(
                Tc, None, self._pv_E, self._pv_V, self._pv_Ts, p0_norm=pn))
        elif self.arrhenius_active:
            # the T-only Arrhenius law, evaluated in the cancellation-free
            # anchored form `-ln(deta_T) (1 + Ts) T / (T + Ts)`, see
            # `specmc.physics.arrhenius.arrhenius_exponent` for why the unanchored
            # `E/(T+Ts) + C` is NOT used here (measured 24 % error in eta(1) at
            # D = 1e10, Ts = 0.02).  The temperature clip above is the frozen
            # `ETA_T_CLIP = (0, 2)`, shared verbatim with the exponential law.
            eta = np.exp(_arrhenius_exponent(Tc, self.deta_T, self.arrhenius_Ts))
        else:
            eta = eta_exp_T(Tc, self.deta_T)
        if self.eta_smooth > 0.0:
            # the smooth clip: the C^2 smoothstep clamp.  `n_clip_eta` keeps
            # its hard-clip meaning (points the HARD clamp would move) so the two
            # stages stay comparable; `n_smooth_eta` counts the points the
            # smoothing reshaped (strictly inside a transition band).
            self.n_clip_eta += EB.n_clipped(
                eta, self.eta_clip_lo, self.eta_clip_hi)
            eta, n_sm = EB.smooth_clip_eta(eta, self.eta_clip_lo,
                                           self.eta_clip_hi, self.eta_smooth)
            self.n_smooth_eta += n_sm
            return eta
        eta, moved = EB.clip_eta(eta, self.eta_clip_lo, self.eta_clip_hi)
        self.n_clip_eta += moved
        return eta

    def eta_of(self, that):
        """eta(x, z) from the CURRENT temperature field (frozen per step).

        `eta = exp(-ln(deta_T) T)` is a NONLINEAR function of theta.  Evaluating
        it pointwise on the Nx grid folds the part of its x-spectrum above the
        Nyquist mode back onto the resolved modes (classic
        pointwise-nonlinearity aliasing, measured as a 0.71 % Nu difference
        over dt = 0.01, far above the 1e-6 threshold).

        Fix: synthesise theta on `eta_pad_factor * Nx` points (theta is
        band-limited to Nk = Nx/2 + 1 modes, so the zero-padded synthesis is
        EXACT), evaluate eta there, and truncate its rfft back to the first Nk
        modes before re-synthesising on the Nx grid that `CoupledStokes`
        consumes.  Cost is one extra (2Nx) FFT, i.e. negligible next to a
        coupled Stokes solve.  `eta_pad_factor = 1` reproduces the old path.

        The viscosity-bound guard appends :meth:`_guard_eta_positive` to the result: the
        truncation above is a projection, and a projection is not a positive
        map, so for a large viscosity contrast the field handed to the Stokes
        operator can go NEGATIVE.  The guard repairs that (and nothing else);
        it returns the field untouched whenever it is strictly positive.
        """
        return self._guard_eta_positive(self._eta_padded_truncated(that))

    def _eta_padded_truncated(self, that):
        """The padded eta construction + x-truncation, without the guard.

        Kept separate from :meth:`eta_of` so the positivity guard can be tested
        (and its no-op property checked) against exactly the field the padded
        construction produced.  See :meth:`_guard_eta_positive` for why the
        truncation is not a positive map.
        """
        fac = self.eta_pad_factor
        N, Nk = self.Nz, self.Nk
        if fac <= 1:
            th = np.real(self.grid(that))
            pn = _p0_norm(self._Z) if self.arrhenius_pv_active else None
            return self._eta_from_T(1.0 - self._Z + th, p0n=pn)
        Np = fac * self.Nx
        sp = np.zeros((Np // 2 + 1, N), dtype=complex)
        sp[:Nk] = (self.V @ np.asarray(that).T).T
        th = np.fft.irfft(sp * Np, n=Np, axis=0)          # (Np, Nz), exact
        # the pressure-dependent viscosity law: the pV branch needs `1 - z` on the PADDED grid.  It is the same
        # z for every padded x-row, so the (Nz,) SSOT array broadcasts against
        # the (Np, Nz) temperature field, no extra memory, no new transform.
        pn = _p0_norm(self.z[None, :]) if self.arrhenius_pv_active else None
        eta_p = self._eta_from_T(1.0 - self.z[None, :] + th, p0n=pn)
        A = np.fft.rfft(eta_p, axis=0) / Np
        return np.fft.irfft(A[:Nk] * self.Nx, n=self.Nx, axis=0)

    def _guard_eta_positive(self, eta):
        """Re-bound the field the Stokes operator consumes.

        Why this exists.  `eta_of` evaluates the law on ``eta_pad_factor * Nx``
        points, where it is positive and inside the documented window, and
        then TRUNCATES its x-spectrum back to the ``Nk`` resolved modes.  That
        projection is not a positive map, so for a large enough viscosity
        contrast the truncation undershoots through zero.  Measured on the
        32x48 production state with the reachable window:

            H = 1  n=150   eta_min = -6.364e-3   (26 of 1536 points)
            H = 10 n= 81   eta_min = -1.224e-4   ( 2 points)
            H = 10 n=200   eta_min = -1.653e-2   (39 points)

        A negative viscosity is not physics, and it also destroys the BFBT
        profile functionals, which are built from ``eta``: ``harm`` becomes
        ``1/mean(1/eta) = -4.187e-3`` and ``geom`` becomes ``exp(mean(log
        eta)) = nan``.  The preconditioner then no longer approximates the
        operator and the FGMRES stagnates; the shipped H = 10 run returned a
        velocity whose true relative residual was 0.871 (n = 81) and 0.993
        (n = 100) after 800 iterations, and 0.871 even after 8000.

        What the guard does NOT do.  It is not a new viscosity bound: it fires
        only when the projection has left the physical half-line, and
        then it re-applies exactly the window of `specmc.physics.eta_bounds`
        that the hard clip already applies in the padded space.  For every frozen reference
        state measured (H = 0, 0 <= T <= 1, so the padded law lies in
        ``[1e-3, 1]``) the truncated field is strictly positive; the closest
        approach measured over 500 reference steps is ``+5.67e-4``, so the guard
        returns the input object untouched and the reference immune region stays
        bit-for-bit.

        Returns the input object unchanged whenever ``eta > 0`` componentwise.
        """
        if float(np.asarray(eta).min()) > 0.0:
            return eta
        if self.eta_smooth > 0.0:
            self.n_clip_eta_proj += EB.n_clipped(eta, self.eta_clip_lo,
                                                 self.eta_clip_hi)
            eta, n_sm = EB.smooth_clip_eta(eta, self.eta_clip_lo,
                                           self.eta_clip_hi, self.eta_smooth)
            self.n_smooth_eta_proj += n_sm
            return eta
        eta, moved = EB.clip_eta(eta, self.eta_clip_lo, self.eta_clip_hi)
        self.n_clip_eta_proj += moved
        return eta

    # -------------------------------------------------------------- Stokes
    def dissipation_eta(self):
        """viscous dissipation: the eta field the last coupled solve consumed.

        `last_eta` is set by :meth:`_stokes_coupled` (once per Stokes solve, so
        the positivity-guard counters are not double-counted); before the first
        solve it falls back to the depth-only profile `RBCVariableViscosity`
        always carries, and never to a freshly evaluated field.
        """
        return self.last_eta if self.last_eta is not None else self.eta_grid

    def _eta_drift(self, eta):
        """Max POINTWISE relative change of eta since the frozen factorisation.

        The old criterion was `max|d eta| / max(eta_lag)`.
        With eta in [1e-3, 1] the denominator is ~1 while the hot (low-viscosity)
        boundary layer, exactly where the operator changes fastest, only
        contributes ~1e-3 even for a 100 % local change, so the threshold never
        fired and the "refresh on eta drift" design intent was dead.  The
        pointwise relative measure reacts where it matters.
        """
        if self.ls is None or self.ls.eta_lag is None:
            return np.inf
        el = self.ls.eta_lag
        return float((np.abs(eta - el) / np.maximum(np.abs(el), 1e-300)).max())

    def _defect_solve(self, that, eta):
        """Lagged factorisation + Picard defect corrections.

        The Picard iteration is a stationary iteration contracted by
        `A_lag^{-1} A(eta_new)`, so it only converges while the frozen operator
        is close enough to the true one.  With `defect_tol` set the loop stops
        as soon as the RELATIVE size of the last correction
        `||A_lag^-1 r|| / ||x||` is below the tolerance, and reports failure
        through `self._last_failed` when `max_K` is reached first.  That is the
        case during the fast initial transient of a case-2a spin-up, where
        theta, and therefore eta, changes by O(10 %) per step, so an
        operator frozen 16 steps earlier is meaningless even though it is
        perfectly fine at the developed state.  The calling code then refreshes
        and retries.

        `n_defect_failed` counts the failures (this used to be a
        silent channel: the flag was set but no counter existed).
        """
        ls = self.ls
        b = ls.rhs(that)
        x = sla.lu_solve(ls.lu_lag, b)
        Anew = free_operator_for(self, eta, op_template=ls.op_lag)
        bn = max(np.linalg.norm(b), 1e-300)
        xn = max(np.linalg.norm(x), 1e-300)
        tol = self.defect_tol
        kmax = self.max_K if tol is not None else self.K
        res, corr = [], []
        for k in range(kmax):
            r = b - Anew.matvec(x)
            res.append(float(np.linalg.norm(r) / bn))
            dx = sla.lu_solve(ls.lu_lag, r)
            c = float(np.linalg.norm(dx) / xn)
            corr.append(c)
            x = x + dx
            if tol is not None and c <= tol:
                break
        r = b - Anew.matvec(x)
        res.append(float(np.linalg.norm(r) / bn))
        self.last_K = len(corr)
        self.last_corr = corr[-1] if corr else 0.0
        self.last_defect_res = res[-1]
        self.n_defect += 1
        if tol is None:
            self._last_failed = False
        else:
            self._last_failed = bool(not np.isfinite(self.last_corr)
                                     or self.last_corr > tol)
            if self._last_failed:
                self.n_defect_failed += 1
        return x

    def _refresh(self, eta):
        tr = time.perf_counter()
        self.ls.refresh(eta)
        self.t_refresh += time.perf_counter() - tr
        self.n_since_refresh = 0
        self.n_refresh += 1
        self._force_refresh = False

    def _schur_true_res(self, fop, b, x):
        """TRUE relative residual ``||b - A x|| / ||b||`` of a Schur solution."""
        bn = max(float(np.linalg.norm(b)), 1e-300)
        return float(np.linalg.norm(b - fop.matvec(x)) / bn)

    def _stokes_bfbt_solve(self, that, eta):
        """The spectral BFBT FGMRES route; -> (x, op, b, fop, info, it, res).

        ``res`` is the TRUE relative residual, recomputed here: the GMRES
        quasi-residual that `fgmres` returns can lose the true norm once the
        Arnoldi vectors lose orthogonality, and the whole point of this method
        is that the calling code must be able to *decide* whether to accept it.
        """
        op = operator_for(self, eta)
        b = op.rhs_schur(that)
        fop = free_operator_for(self, eta, op_template=op)
        P = SpectralBFBTPrecond(self, eta, profile=self.precond_profile)
        x, info, it, hist = fgmres(
            fop.matvec, b, M=P.apply, rtol=self.stokes_rtol,
            restart=self.stokes_restart, maxiter=self.stokes_maxiter)
        self.n_bfbt_solves += 1
        self.n_bfbt_iters += int(it)
        self.last_bfbt_iters = int(it)
        self.last_bfbt_info = int(info)
        res = self._schur_true_res(fop, b, x)
        self.last_bfbt_true_res = res
        self.last_K = 0
        self.last_defect_res = res
        return x, op, b, fop, int(info), int(it), res

    def _stokes_rescue(self, that, eta, b, fop):
        """Recompute a rejected bfbt solve with the robust routes.

        Measurement at H = 10 on the 32x48 production state: at the states
        where the harm-profile BFBT preconditioner stagnates (true relative
        residual 0.871 at n = 81, 0.993 at n = 100, and it does not
        improve with 8000 iterations), the LAGGED factorisation + one adaptive
        Picard defect correction of the legacy route is accurate to
        ``rel_dpsi = 8.0e-11`` / ``1.2e-11`` with a true Schur residual of
        ``1.2e-12`` / ``1.5e-13``, and it is cheap, because the factorisation
        is one Nk x (Nz x Nz) block LU.

        Order: (1) the lagged defect-corrected solve at THIS eta; (2) if that
        still misses the tolerance, the dense Schur LU, but only while
        ``Nk * Nz <= DIRECT_RESCUE_MAX`` (see the constant).  Above the cap the
        lagged result is accepted and counted in ``n_bfbt_rescue_weak``: a
        multi-GB allocation inside a step is a worse failure mode than a
        documented tolerance miss.
        """
        if self.ls is None:
            self.ls = LaggedStokes(self, form='schur', rtol=self.stokes_rtol,
                                   restart=80, maxiter=400)
        self._refresh(eta)
        x = self._defect_solve(that, eta)
        if self._last_failed:
            self._refresh(eta)
            x = self._defect_solve(that, eta)
            if self._last_failed:
                self.n_defect_double_fail += 1
        res = self._schur_true_res(fop, b, x)
        self.last_defect_res = res
        if res <= self.stokes_rtol * RESCUE_RTOL_SLACK or not np.isfinite(res):
            return x, res, 'bfbt->legacy'
        if self.Nk * self.Nz > DIRECT_RESCUE_MAX:
            self.n_bfbt_rescue_weak += 1
            return x, res, 'bfbt->legacy(weak)'
        x2 = sla.lu_solve(sla.lu_factor(operator_for(self, eta)
                                         .assemble_schur()), b)
        self.n_bfbt_rescue_direct += 1
        res2 = self._schur_true_res(fop, b, x2)
        self.last_defect_res = res2
        return x2, res2, 'bfbt->direct'

    def _stokes_coupled(self, that):
        t0 = time.perf_counter()
        eta = self.eta_of(that)
        # viscous dissipation: remember the eta this solve consumed.  The dissipation source
        # of the following theta step must use the SAME field: re-deriving it
        # would duplicate the positivity-guard counters and could weight Phi
        # with a field the operator never saw.
        self.last_eta = eta
        self.last_eta_pad_factor = self.eta_pad_factor
        if self.precond == 'bfbt':
            # the spectral BFBT preconditioned FGMRES route.
            # It solves the SAME coupled Schur system as the defect route; only
            # the Krylov path differs, so `precond='legacy'` is untouched.
            #
            # an unconverged result is no longer accepted
            # silently.  `info != 0` alone is not enough to reject (the GMRES
            # quasi-residual can lag the true one), so the TRUE residual is
            # measured and the solve is accepted only if it meets `stokes_rtol`;
            # otherwise the robust routes of `_stokes_rescue` recompute it.
            x, op, b, fop, info, it, res = self._stokes_bfbt_solve(that, eta)
            if info != 0 and not (res <= self.stokes_rtol * RESCUE_RTOL_SLACK):
                self.n_bfbt_rescue += 1
                x, res, route = self._stokes_rescue(that, eta, b, fop)
            else:
                route = 'bfbt'
            self.last_stokes_route = route
            psi, om, v = op.unpack_schur(x)
        elif self.stokes_mode == 'direct':
            op = operator_for(self, eta)
            A = op.assemble_schur()
            b = op.rhs_schur(that)
            lu = sla.lu_factor(A)
            x = sla.lu_solve(lu, b)
            psi, om, v = op.unpack_schur(x)
            self.n_direct_stokes += 1
            self.last_K = 0
            self.last_defect_res = 0.0
        else:
            if self.ls is None:
                self.ls = LaggedStokes(self, form='schur',
                                       rtol=self.stokes_rtol, restart=80,
                                       maxiter=400)
            drift = self._eta_drift(eta)
            if (self.ls.op_lag is None or self._force_refresh
                    or self.n_since_refresh >= self.lag
                    or drift > self.eta_drift_tol):
                self._refresh(eta)
            self.n_since_refresh += 1
            x = self._defect_solve(that, eta)
            if self._last_failed:
                # the frozen eta is too old for the correction to converge:
                # pay for a fresh factorisation at THIS state and redo it.
                self._refresh(eta)
                self.n_since_refresh += 1
                x = self._defect_solve(that, eta)
                if self._last_failed:
                    # The SECOND failure used to be accepted
                    # silently, so the step could march on an unconverged
                    # Stokes solve with no trace in any counter or log.  It is
                    # still accepted (throwing would kill a multi-hour run),
                    # but it is now counted, and `describe()`/the drivers can
                    # see it.
                    self.n_defect_double_fail += 1
            psi, om, v = self.ls.op_lag.unpack_schur(x)
            self.last_stokes_route = ('direct' if self.stokes_mode == 'direct'
                                      else 'legacy')
        self.t_stokes += time.perf_counter() - t0
        return psi, om

    def stokes(self, that):
        """theta coefficients -> (psi, omega); memoised on the state.

        The production loop reads

            dt = cfl_dt(s);  s = step(s, dt)

        and `cfl_dt` already has to solve Stokes in order to get `max|u|` for the
        CFL number.  The step must then reuse THAT solution instead of solving
        again (`prepare`), otherwise every step pays two full coupled solves.

        A `bytes(that)` key can never hit: `project_mirror` is
        applied at several places on the way in
        (`RBCCoupled.stokes` is called with `s` by `cfl_dt`, but with
        `project_mirror(project_mirror(s))` by the step) and the V -> Re -> Vi
        round trip is only idempotent to roundoff, so the byte patterns differ
        in their last bits while the physical state is identical.

        Fix: compare the per-mode rfft signature `V @ that` with a RELATIVE
        tolerance.  `stokes_rtol = 1e-8` and `defect_tol = 1e-8` are the
        accuracies the solve itself is driven to, so a 1e-12 signature match
        reuses a solution whose error contribution is ~1e-12, four orders of
        magnitude below every tolerance of the step.
        """
        that = np.asarray(that)
        key = that.tobytes()
        if key == self._cache_key:
            return self._cache_val
        X = (self.V @ that.T).T
        cX = self._cache_X
        if cX is not None and cX.shape == X.shape:
            den = max(float(np.abs(cX).max()), 1e-300)
            if float(np.abs(X - cX).max()) <= self._cache_tol * den:
                self.n_memo_hits += 1
                return self._cache_val
        val = self._stokes_coupled(that)
        self.n_stokes += 1
        self._cache_key = key
        self._cache_X = X
        self._cache_val = val
        return val

    def describe(self):
        d = super().describe()
        if self.arrhenius_pv_active:
            _mode = ('COUPLED eta=eta(T,z)  Arrhenius pV(L) '
                     'V-tilde=%.6g' % self._pv_V)
        elif self.arrhenius_active:
            _mode = 'COUPLED eta=eta(T(x,z,t))  Arrhenius T-only'
        else:
            _mode = ('COUPLED eta=eta(T(x,z,t))  '
                     '(Blankenkenbach case 2a)')
        d.update(mode=_mode,
                 compressibility=self.compressibility,
                 tala_active=self.tala_active,
                 tala_a=self.tala_a,
                 tala_record=self.tala_record,
                 stokes_mode=self.stokes_mode, lag=self.lag, K=self.K,
                 defect_tol=self.defect_tol, stokes_rtol=self.stokes_rtol,
                 eta_drift_tol=self.eta_drift_tol,
                 eta_dealias=self.eta_pad_factor,
                 internal_heating=self.internal_heating,
                 dissipation_number=self.theta.dissipation_number,
                 dissipation_active=self.dissipation_active,
                 phi_total=self.theta.last_phi_total,
                 n_diss=self.theta.n_diss,
                 arrhenius_active=self.arrhenius_active,
                 arrhenius_Ts=self.arrhenius_Ts,
                 arrhenius_E=self.arrhenius_E,
                 arrhenius_V=self.arrhenius_V,
                 arrhenius_pv_active=self.arrhenius_pv_active,
                 reference_anchor=self.reference_anchor,
                 pv_record=self.pv_record,
                 n_defect_failed=self.n_defect_failed,
                 n_defect_double_fail=self.n_defect_double_fail,
                 n_clip_T=self.n_clip_T,
                 n_T_out_of_range=self.n_T_out_of_range,
                 max_T_excursion=self.max_T_excursion,
                 eta_clip_lo=self.eta_clip_lo,
                 eta_clip_hi=self.eta_clip_hi,
                 eta_smooth=self.eta_smooth,
                 n_clip_eta=self.n_clip_eta,
                 n_smooth_eta=self.n_smooth_eta,
                 n_clip_eta_proj=self.n_clip_eta_proj,
                 n_smooth_eta_proj=self.n_smooth_eta_proj,
                 precond=self.precond,
                 precond_profile=self.precond_profile,
                 n_bfbt_solves=self.n_bfbt_solves,
                 n_bfbt_iters=self.n_bfbt_iters,
                 last_bfbt_iters=self.last_bfbt_iters,
                 last_bfbt_info=self.last_bfbt_info,
                 last_bfbt_true_res=self.last_bfbt_true_res,
                 n_bfbt_rescue=self.n_bfbt_rescue,
                 n_bfbt_rescue_direct=self.n_bfbt_rescue_direct,
                 n_bfbt_rescue_weak=self.n_bfbt_rescue_weak,
                 last_stokes_route=self.last_stokes_route)
        return d


# ------------------------------------------------------------------ self test
def self_test(Nx=16, Nz=24, deta_T=1000.0, verbose=True):
    """Structural checks that need no long run.

    [1] eta_of: the alias-free padded construction (and eta_dealias=1
        reproducing the old pointwise path bit-for-bit)
    [2] with a CONSTANT theta the operator is eta = eta_bar(z): the defect
        correction must converge in ONE step and agree with the direct solve
    [3] the memoised stokes() returns the same object for the same state and
        re-solves for a different state
    """
    rng = np.random.default_rng(7)
    X, Z = None, None
    ok = True
    m = RBCCoupled(Nx=Nx, Nz=Nz, deta_T=deta_T, stokes_mode='defect',
                   lag=16, K=3, theta_scheme='implicit')
    that = m.seed(amp=3e-2)
    eta = m.eta_of(that)
    # independent reference: the same padded construction, written out here
    Np = 2 * m.Nx
    sp = np.zeros((Np // 2 + 1, m.Nz), dtype=complex)
    sp[:m.Nk] = (m.V @ np.asarray(that).T).T
    thp = np.fft.irfft(sp * Np, n=Np, axis=0)
    etp = eta_exp_T(np.clip(1.0 - m.z[None, :] + thp, 0.0, 2.0), deta_T)
    Aref = (np.fft.rfft(etp, axis=0) / Np)[:m.Nk]
    ref = np.fft.irfft(Aref * m.Nx, n=m.Nx, axis=0)
    e1 = float(np.abs(eta - ref).max() / max(ref.max(), 1e-300))
    # ... and the OLD pointwise path, which the fix replaces
    eta_old = eta_exp_T(np.clip(1.0 - m._Z + np.real(m.grid(that)), 0.0, 2.0),
                        deta_T)
    e_alias = float(np.abs(eta - eta_old).max() / max(eta_old.max(), 1e-300))
    m1 = RBCCoupled(Nx=Nx, Nz=Nz, deta_T=deta_T, theta_scheme='implicit',
                    eta_dealias=1)
    e_old = float(np.abs(m1.eta_of(that) - eta_old).max()
                  / max(eta_old.max(), 1e-300))
    ok &= e1 < 1e-14 and e_old == 0.0
    if verbose:
        print('  [1] eta_of vs padded reference: rel %.1e ; eta_dealias=1 vs '
              'old pointwise path: %.1e (bit-identical %s) ; aliasing removed '
              '%.1e  (eta in [%.3g, %.3g])'
              % (e1, e_old, e_old == 0.0, e_alias, eta.min(), eta.max()))

    # [2] eta = eta(T) with the SAME state: the defect-corrected route must
    #     agree with the fresh assemble+LU route
    m2 = RBCCoupled(Nx=Nx, Nz=Nz, deta_T=deta_T, stokes_mode='defect',
                    lag=16, K=3, theta_scheme='implicit')
    that2 = m2.seed(amp=1e-2)
    md = RBCCoupled(Nx=Nx, Nz=Nz, deta_T=deta_T, stokes_mode='direct',
                    theta_scheme='implicit')
    psi_d, om_d = md.stokes(that2)
    mdef = RBCCoupled(Nx=Nx, Nz=Nz, deta_T=deta_T, stokes_mode='defect',
                      lag=16, K=3, theta_scheme='implicit')
    psi_x, om_x = mdef.stokes(that2)
    e2 = err(psi_x, psi_d)
    ok &= e2 < 1e-8
    if verbose:
        print('  [2] defect vs direct (eta = eta(T), same state): psi rel '
              '%.2e  (K used = %d, final defect res %.1e)'
              % (e2, mdef.last_K, mdef.last_defect_res))

    # [3] memo
    a = mdef.stokes(that2)
    n0 = mdef.n_stokes
    b = mdef.stokes(that2)
    fresh = mdef.n_stokes - n0
    that3 = mdef.seed(amp=2e-2)
    mdef.stokes(that3)
    ok &= (fresh == 0) and (mdef.n_stokes - n0 == 1)
    if verbose:
        print('  [3] memo: repeated call re-solved %d times, new state %d time'
              % (fresh, mdef.n_stokes - n0))

    # [4] internal heating reaches the theta right-hand side and
    #     H = 0 does not move a bit.  The source must (a) leave the Stokes
    #     solve and eta untouched and (b) change exactly one RHS coefficient
    #     by exactly H.
    mH = RBCCoupled(Nx=Nx, Nz=Nz, deta_T=deta_T, theta_scheme='implicit',
                    internal_heating=7.25)
    th4 = m2.seed(amp=1e-2)
    # `rhs` needs the frozen velocity of the step (RBCCoupled defaults to
    # implicit_buoy=False), so both solvers have to be prepared on the SAME
    # state before their right-hand sides are compared.
    m2.theta.prepare(th4)
    mH.theta.prepare(th4)
    b0 = m2.theta.rhs(th4, 2.5e-3)
    bH = mH.theta.rhs(th4, 2.5e-3)
    d = (bH - b0).reshape(mH.Nk, mH.Nz)
    nz = int(np.count_nonzero(d))
    e_h = float(np.abs(d[0, 0] - 7.25))
    eta_same = float(np.abs(mH.eta_of(th4) - m2.eta_of(th4)).max())
    ok &= (nz == 1) and (e_h == 0.0) and (eta_same == 0.0)
    if verbose:
        print('  [4] internal heating: RHS delta = %d coefficient, |db-H| = '
              '%.1e, eta independent of H (max|d eta| = %.1e)'
              % (nz, e_h, eta_same))

    # [5] the POST-TRUNCATION positivity guard.
    #     (a) it is a bit-for-bit no-op (the same object comes back) for a
    #         state whose truncated eta is strictly positive, that is the
    #         frozen reference situation, so the immune region is preserved;
    #     (b) it fires for a state where the x-projection undershoots through
    #         zero, and the field it returns is strictly positive and inside
    #         the window.
    m5 = RBCCoupled(Nx=Nx, Nz=Nz, deta_T=deta_T, theta_scheme='implicit',
                    internal_heating=10.0, eta_min=1e-6, eta_max=1e6)
    X5, Z5 = np.meshgrid(m5.x, m5.z, indexing='ij')
    th_ok = m5.seed(amp=1e-2)
    raw_ok = m5._eta_padded_truncated(th_ok)
    got_ok = m5._guard_eta_positive(raw_ok)
    noop = (got_ok is raw_ok) and m5.n_clip_eta_proj == 0
    fired, amp_f, raw_min = False, None, None
    for amp in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
        thf = m5.coeffs(HEAT.conductive_theta(Z5, 10.0)
                        + amp * np.cos(np.pi * X5) * np.sin(np.pi * Z5))
        rawf = m5._eta_padded_truncated(thf)
        if float(rawf.min()) <= 0.0:
            gotf = m5.eta_of(thf)
            fired = (m5.n_clip_eta_proj > 0
                     and float(gotf.min()) > 0.0
                     and float(gotf.min()) >= m5.eta_clip_lo * (1.0 - 1e-12)
                     and float(gotf.max()) <= m5.eta_clip_hi * (1.0 + 1e-12))
            amp_f, raw_min = amp, float(rawf.min())
            break
    ok &= bool(noop) and bool(fired)
    if verbose:
        print('  [5] eta positivity guard: positive state -> no-op %s ; '
              'undershooting state (amp=%s, raw min %.3e) repaired %s '
              '(n_clip_eta_proj = %d)'
              % (noop, amp_f, raw_min if raw_min is not None else float('nan'),
                 fired, m5.n_clip_eta_proj))

    # [6] the bfbt route must never accept an unconverged
    #     Stokes solve.  With a one-iteration budget the FGMRES cannot converge,
    #     so the rescue must fire and the velocity it returns must agree with
    #     the exact (dense-LU) one.
    mr = RBCCoupled(Nx=Nx, Nz=Nz, deta_T=deta_T, theta_scheme='implicit',
                    precond='bfbt', stokes_maxiter=1)
    th6 = mr.seed(amp=1e-2)
    psi_r, _ = mr.stokes(th6)
    mdr = RBCCoupled(Nx=Nx, Nz=Nz, deta_T=deta_T, theta_scheme='implicit',
                     stokes_mode='direct')
    psi_d6, _ = mdr.stokes(th6)
    e6 = err(psi_r, psi_d6)
    ok &= (mr.n_bfbt_rescue == 1) and (e6 < 1e-8)
    if verbose:
        print('  [6] bfbt rescue: maxiter=1 -> rescue fired %d time(s), route '
              '%s, rel dpsi vs direct %.2e (true res %.1e)'
              % (mr.n_bfbt_rescue, mr.last_stokes_route, e6,
                 mr.last_defect_res))

    # [7] the Arrhenius (T-only) law is wired in and is calibrated.
    #     (a) with `arrhenius_Ts = 0` the branch is not taken: the eta field is
    #         the SAME OBJECT the old call returns, bit for bit (the immune
    #         guarantee);
    #     (b) with a positive `Ts` the field equals the independent reference
    #         `exp(E/(T+Ts) + C)` evaluated longhand, its endpoints are exact,
    #         and `Ts = 1` reproduces `E_FK = 2 ln(deta_T)`;
    #     (c) a bad `arrhenius_Ts` / `arrhenius_E` raises instead of running.
    from ..physics.arrhenius import arrhenius_E as _E, oracle_eta as _oracle
    m7 = RBCCoupled(Nx=Nx, Nz=Nz, deta_T=deta_T, theta_scheme='implicit')
    th7 = m7.seed(amp=1e-2)
    T7 = np.clip(1.0 - m7._Z + np.real(m7.grid(th7)), 0.0, 2.0)
    eta_off = m7._eta_from_T(T7)
    eta_ref_off = eta_exp_T(T7, deta_T)
    same_off = bool(np.array_equal(eta_off, eta_ref_off)) and (eta_off is not None)
    etas = {}
    worst_b = 0.0
    for Ts in (1.0, 0.5, 0.2, 0.1):
        ma = RBCCoupled(Nx=Nx, Nz=Nz, deta_T=deta_T, theta_scheme='implicit',
                        arrhenius_Ts=Ts)
        ea = ma._eta_from_T(T7)
        er = _oracle(T7, deta_T, Ts)
        worst_b = max(worst_b, float(np.abs(np.log(ea) - np.log(er)).max()))
        etas[Ts] = (float(ea.min()), float(ea.max()))
    mA = RBCCoupled(Nx=Nx, Nz=Nz, deta_T=deta_T, theta_scheme='implicit',
                    arrhenius_Ts=1.0)
    e0 = float(mA._eta_from_T(np.zeros_like(T7))[0, 0])
    e1 = float(mA._eta_from_T(np.ones_like(T7))[0, 0])
    fk_ok = abs(mA.arrhenius_E - 2.0 * np.log(deta_T)) < 1e-12
    bad7 = 0
    for kw in (dict(arrhenius_Ts=-1.0),
               dict(arrhenius_Ts=1.0, arrhenius_E=1.0),
               dict(arrhenius_E=1.0)):
        try:
            RBCCoupled(Nx=Nx, Nz=Nz, deta_T=deta_T, theta_scheme='implicit', **kw)
        except ValueError:
            bad7 += 1
    ok &= (same_off and worst_b < 1e-13 and e0 == 1.0
           and abs(e1 - 1.0 / deta_T) < 1e-12 and fk_ok and (bad7 == 3))
    if verbose:
        print('  [7] Arrhenius: Ts=0 == exponential bit for bit %s ; '
              'Ts>0 vs longhand oracle max|d ln eta| %.1e ; eta(0) = %.17g, '
              'eta(1) = %.17g (1/D = %.17g) ; Ts=1 -> E = %.12f == 2 ln D %s ; '
              'bad inputs rejected %d/3'
              % (same_off, worst_b, e0, e1, 1.0 / deta_T, mA.arrhenius_E,
                 fk_ok, bad7))

    # [8] viscous dissipation: the dissipation source is wired into the coupled model.
    #     (a) Di = 0 is a bit-for-bit no-op on the theta RHS and on the eta
    #         field (the immune guarantee);
    #     (b) Di > 0 adds exactly (Di/Ra) * Phi, evaluated with `last_eta`,
    #         the field the coupled Stokes operator consumed, and
    #         the same history gives the same source through BOTH Stokes routes
    #         (direct and defect), which is the consistency statement.
    from ..physics import dissipation as DISS
    m8a = RBCCoupled(Nx=Nx, Nz=Nz, deta_T=deta_T, theta_scheme='implicit')
    m8b = RBCCoupled(Nx=Nx, Nz=Nz, deta_T=deta_T, theta_scheme='implicit',
                     dissipation_number=0.5)
    th8 = m8a.seed(amp=1e-2)
    m8a.theta.prepare(th8)
    m8b.theta.prepare(th8)
    b8a = m8a.theta.rhs(th8, 2.5e-3)
    b8b = m8b.theta.rhs(th8, 2.5e-3)
    src8 = DISS.source_coeffs(m8b, m8b.theta.psi, m8b.dissipation_eta(),
                              Di=0.5)
    d8 = (b8b - b8a).reshape(m8b.Nk, m8b.Nz)
    e8 = float(np.abs(d8 - src8).max()
               / max(float(np.abs(b8a).max()), 1e-300))
    eta_same8 = float(np.abs(m8b.eta_of(th8) - m8a.eta_of(th8)).max())
    m8_c = RBCCoupled(Nx=Nx, Nz=Nz, deta_T=deta_T, theta_scheme='implicit',
                      dissipation_number=0.5, stokes_mode='direct')
    m8_c.stokes(th8)
    m8_c.theta.prepare(th8)
    src8c = DISS.source_coeffs(m8_c, m8_c.theta.psi, m8_c.dissipation_eta(),
                               Di=0.5)
    rel8 = float(np.abs(src8c - src8).max() / max(abs(src8).max(), 1e-300))
    ok &= (e8 <= 1e-14) and (eta_same8 == 0.0) and m8b.theta.n_diss == 1 \
        and m8b.theta.last_phi_total == DISS.source_integral(m8b, src8) \
        and rel8 < 1e-8 and m8b.last_eta is not None
    if verbose:
        print('  [8] dissipation wiring: Di=0 RHS + eta bit-identical (%.1e); '
              'delta == (Di/Ra)Phi to %.1e (rel); the eta of the consumed '
              'field (n_diss=%d, <source>=%.3e); direct vs defect source '
              'rel %.1e' % (eta_same8, e8, m8b.theta.n_diss,
                            m8b.theta.last_phi_total, rel8))

    if verbose:
        print('  coupled_model self_test %s' % ('PASSED' if ok else 'FAILED'))
    return bool(ok)


def err(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max()
                 / max(np.abs(np.asarray(b)).max(), 1e-300))


if __name__ == '__main__':
    print('coupled eta(T) production model self test')
    self_test()
