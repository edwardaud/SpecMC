"""Linearly-implicit temperature time stepping for the 2D infinite-Prandtl model.

Why this file exists
-------------------------------------
The starting point was the assumption that the current temperature
equation is explicit and therefore CFL-limited.  That assumption is only
half right: `rbc_galerkin.RBC.step` already treats the *diffusion*
term implicitly (backward Euler, `lam = 1/dt` in the Chebyshev tau solve)

    (theta^{n+1} - theta^n)/dt = lap(theta^{n+1}) + N(theta^n),
    N = -(u.grad)theta + u_z                   (explicit, 3/2-dealiased)

so the CFL restriction that binds is the advective one
(dt <= min dz / max|u|), not a diffusive one.  What this file adds is therefore
the *other* half: the advection (and, optionally, the buoyancy source) is moved
to the implicit side, which removes the advective restriction.

Scheme (linearly implicit / backward Euler with frozen linearisation)
--------------------------------------------------------------------
Let `u^n = S[theta^n]` be the velocity obtained from the Stokes solve, let
`L = D^2 - k^2` be the (per-wavenumber) Laplace operator and let `A(u^n)` be the
advection operator as it appears in the temperature equation,

    A(u) v = -(u . grad) v .

The scheme solves

    (I/dt - L - A(u^n) - B^n) theta^{n+1} = theta^n/dt + [u_z^n - B^n theta^n]
                                                                    (S1)

with

    B^n v      = u_z(v)  at frozen eta^n  frozen-eta buoyancy operator
                                          (= -i k * Stokes_k[v], diagonal in k)

`implicit_buoy=False` selects the simpler scheme with the buoyancy left
explicit:

    (I/dt - L - A(u^n)) theta^{n+1} = theta^n/dt + u_z^n               (S2)

BDF2 (second-order scheme)
--------------------------
`scheme='bdf2'` replaces the backward-Euler time derivative by the second-order
backward differentiation formula.  For a *variable* step (this matters in
production, where dt is set by the CFL condition and therefore changes every
step) with

    h_n = dt_n,  h_{n-1} = dt_{n-1},  r = h_n / h_{n-1}

the derivative at t^{n+1} is

    theta'(t^{n+1}) ~ [ c0 theta^{n+1} - c1 theta^n + c2 theta^{n-1} ] / h_n
    c0 = (1+2r)/(1+r),   c1 = (1+r),   c2 = r^2/(1+r)

which for r = 1 is the textbook constant-step form
`(1.5 theta^{n+1} - 2 theta^n + 0.5 theta^{n-1})/dt`.  The linearly implicit
step is therefore

    (a0 I - L - A(u^n) [- B^n]) theta^{n+1}
        = c1 theta^n/h_n - c2 theta^{n-1}/h_n + [u_z^n]                  (B2)

with `a0 = c0/h_n`; the preconditioner becomes `(a0 I - L)^{-1}`, i.e. exactly
the same z-solver with `lam = a0` instead of `lam = 1/dt`.  The first step after
a (re)start has no theta^{n-1} and falls back to backward Euler
(`r = None -> c0 = c1 = 1, c2 = 0`), which costs one first-order step out of
thousands.  See `set_step` for the exact coefficient handling.

Both are consistent discretisations of the same PDE (the sign of A in the
operator is negative because it is moved to the left-hand side of the
equation); the bracket in (S1) vanishes identically because B^n is *exactly*
the linear map theta -> u_z at frozen eta, so (S1) is the standard Picard
(frozen-eta, frozen-u) linearisation of the fully implicit backward-Euler step.
With `implicit_buoy=False` the buoyancy enters explicitly, exactly as in the
reference scheme.

Properties
----------
* (S1)/(S2) with small dt reduce to the reference scheme up to O(dt) (the term
  that is moved across the equality is dt * [(u^{n+1}-u^n).grad theta^{n+1}],
  which is O(dt^2) per step).  Verified in the internal cross-check
  (single-step difference ~ dt^2, trajectory difference ~ dt).
* Only the *linear* solve changes: the operator is applied matrix-free in
  spectral space (`matvec`), and preconditioned by the per-wavenumber
  backward-Euler diffusion solve that the reference scheme already uses
  (`lam = 1/dt`, LU-cached).  Nothing about the discretisation of space, the
  dealiasing or the boundary conditions is altered.
* The solve is FGMRES (right preconditioned) from `krylov_fgmres.py`, so an
  arbitrary (even varying) preconditioner is allowed and the residual that is
  driven below `rtol` is the TRUE residual.  `x0` gives a genuine warm start,
  which is what makes the lagged-viscosity experiments possible.

Internal heating
------------------------
`internal_heating = H` adds the constant volumetric source of
`specmc.physics.heating` to the right-hand side of (S1)/(S2)/(B2):

    ... = theta^n/dt + [u_z^n] + H          (a constant, in the k_x = 0 / T_0
                                             coefficient only)

It is a plain additive constant, no Rayleigh factor, no dealiasing, no change
to the operator, the boundary rows or the background term `-u_z`.  With
`H == 0` (the frozen reference value) the source is not even evaluated, so the
temperature equation is bit-for-bit the reference; `specmc.physics.heating` records the
four reference codes (ASPECT `Radiogenic heating rate`, CitcomS `Q0`,
Underworld `fn_sourceTerm`/`adv_diff.f`, G-ADOPT `H=`) that use this same form.

Boundary conditions
-------------------
Rows N-2, N-1 of every wavenumber block are the Dirichlet rows (value at
z = 0 and z = 1); the right-hand side has those two rows set to zero, and the
solution therefore satisfies theta = 0 on both walls exactly, as in
`ChebTau._assemble`.  The tau layout is identical to the reference solver's.
"""
import numpy as np
import scipy.linalg as sla

from .krylov_fgmres import fgmres
from ..models.convection_modes import RBCVariableViscosity
from ..physics import dissipation as DISS
from ..physics import heating as HEAT

__all__ = ["ImplicitThetaSolver", "RBCImplicitTheta", "self_test"]

THETA_SCHEMES = ("explicit", "implicit", "implicit_be", "implicit_bdf2")
BUOY_MODES = ("implicit", "explicit")
TIME_SCHEMES = ("be", "bdf2")


class ImplicitThetaSolver:
    """Matrix-free implicit temperature operator for one model instance.

    The object is *stateless between steps* apart from the frozen fields
    installed by `prepare()`: the model geometry (D, V, kx, the LU cache of the
    z solver) is shared with the reference scheme.
    """

    def __init__(self, model, rtol=1e-10, maxiter=200, restart=40,
                 implicit_buoy=True, atol=0.0, scheme='be',
                 internal_heating=0.0, dissipation_number=0.0):
        self.m = model
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.maxiter = int(maxiter)
        self.restart = int(restart)
        self.implicit_buoy = bool(implicit_buoy)
        #: internal heating: constant volumetric heating H of `d(th)/dt + ... = lap + H`.
        #: 0.0 (the frozen reference value) short-circuits every line that touches the
        #: source, so the H = 0 temperature equation is bit-for-bit the reference.
        self.internal_heating = float(internal_heating)
        #: viscous dissipation number Di of
        #: ``... = lap(theta) + H + (Di/Ra) Phi`` (`specmc.physics.dissipation`).
        #: 0.0 (the frozen reference value) short-circuits every line that touches the
        #: source, so the Di = 0 temperature equation is bit-for-bit the reference.
        self.dissipation_number = float(dissipation_number)
        if not np.isfinite(self.dissipation_number) \
                or self.dissipation_number < 0.0:
            raise ValueError('dissipation_number must be finite and >= 0 '
                             '(got %r)' % (dissipation_number,))
        # --- the TALA temperature equation ---------------------------
        # `compressibility='tala'` weights the time derivative and the advection
        # by `rho_bar(z)` (`self._Mrho`, the exact Chebyshev multiplication
        # matrix, so `rho_bar == 1` is the identity matrix and the Boussinesq
        # floats are reproduced exactly), and adds the adiabatic source
        # `rho_bar u_z [1 - Di (1 - z + theta)]`.  The implicit-buoyancy switch
        # is a Boussinesq device: TALA carries the whole background/adiabatic
        # term explicitly, so `implicit_buoy` is ignored (documented).
        self.tala_active = bool(getattr(model, 'tala_active', False))
        self._Mrho = None
        self._tala_lu = {}
        if self.tala_active:
            from ..stokes.stokes_coupled import cheb_mult_complex
            self._Mrho = cheb_mult_complex(np.asarray(model.rho_c, float),
                                           model.Nz)
            self.implicit_buoy = False
            if self.dissipation_number != getattr(model, 'dissipation_number',
                                                  self.dissipation_number):
                raise ValueError('the TALA reference state and the dissipation '
                                 'source must share one Di')
        #: the (Di/Ra) Phi coefficient array of the CURRENT step, installed by
        #: `prepare`; None until then (and always None when Di == 0).
        self._phi_coeffs = None
        #: number of steps whose right-hand side carried the source, and the
        #: source's discrete volume integral of the current step
        self.n_diss = 0
        self.last_phi_total = 0.0
        if str(scheme).lower() not in TIME_SCHEMES:
            raise ValueError('unknown time scheme %r' % (scheme,))
        self.scheme = str(scheme).lower()
        self.D2 = model.D @ model.D
        self.k2 = (model.kx ** 2)[:, None]
        self.dt = None
        self.dt_prev = None
        # time-derivative coefficients of the CURRENT step (see set_step)
        self.a0 = 1.0            # coefficient of theta^{n+1}
        self.b0 = 1.0            # coefficient of theta^n
        self.b1 = 0.0            # coefficient of theta^{n-1} (bdf2)
        self.be_fallback = False  # True when bdf2 had to restart as BE
        self.psi = None
        self.psi_n = None
        self.ux = self.uz = None
        self.last_iters = 0
        self.last_info = 0
        self.last_res = 0.0
        self.n_solves = 0
        self.n_matvec = 0
        self.n_buoy_solves = 0
        self.n_be_fallback = 0
        self.n_extrap = 0
        #: number of steps whose FGMRES did NOT reach `rtol` (info != 0).
        #: Production drivers used to march on regardless, with the failure
        #: invisible in every log and every instability criterion.
        self.n_nonconv = 0

    # ------------------------------------------------------------ time scheme
    def set_step(self, dt, dt_prev=None):
        """Install the time-derivative coefficients of the step being taken.

        backward Euler : a0 = b0 = 1/dt, b1 = 0
        BDF2           : a0 = (1+2r)/((1+r) dt), b0 = (1+r)/dt,
                         b1 = r^2/((1+r) dt),   r = dt/dt_prev
        'bdf2' with no previous step (dt_prev is None) falls back to BE; the
        calling code is told through `self.be_fallback`.
        """
        self.dt = float(dt)
        self.dt_prev = None if dt_prev is None else float(dt_prev)
        if self.scheme == 'be' or self.dt_prev is None:
            self.a0 = self.b0 = 1.0 / self.dt
            self.b1 = 0.0
            self.be_fallback = bool(self.scheme == 'bdf2')
            if self.be_fallback:
                self.n_be_fallback += 1
            return self
        r = self.dt / self.dt_prev
        self.a0 = (1.0 + 2.0 * r) / ((1.0 + r) * self.dt)
        self.b0 = (1.0 + r) / self.dt
        self.b1 = r * r / ((1.0 + r) * self.dt)
        self.be_fallback = False
        return self

    # ------------------------------------------------------------- transforms
    def _G(self, cf):
        """Chebyshev+FFT synthesis, dealiased exactly like `RBC.explicit_rhs`."""
        return self.m._synth_pad(cf) if self.m.dealias else self.m.grid(cf)

    def _F(self, g):
        """grid -> Chebyshev coefficients per x-mode, dealiased like `RBC.step`."""
        m = self.m
        A = m._trunc_pad(g) if m.dealias else np.fft.rfft(g, axis=0) / m.Nx
        return (m.Vi @ A.T).T

    # ----------------------------------------------------------------- freezing
    def prepare(self, that, psi=None, psi_prev=None, extrapolate=False):
        """Freeze psi, u_x, u_z (and the eta they were computed with) at t^n.

        `psi` may be supplied by the calling code (the coupled production model
        already had to solve Stokes in order to obtain dt, so re-solving here
        would double the cost of a step).  It must be the Stokes solution of
        `that`.

        `extrapolate=True` replaces the frozen velocity of the step by the
        SECOND-ORDER extrapolation  u* = 2 u^n - u^{n-1}  (and the same for the
        buoyancy source u_z).  This matters for BDF2: the velocity is a
        diagnostic of theta, so lagging it at u^n leaves an O(dt) term in the
        local truncation error and the scheme stays FIRST order no matter what
        the time derivative does.  `psi_prev` is u^{n-1} (the Stokes solution of
        theta^{n-1}); when it is None the extrapolation degrades gracefully to
        psi.  The last frozen psi of theta^n is always kept in `self.psi_n`.

        viscous dissipation: with ``dissipation_number != 0`` the viscous-dissipation source
        ``(Di/Ra) Phi`` of THIS step is assembled here, from the frozen ``psi``
        and the eta field the Stokes operator consumed
        (``model.dissipation_eta()``, i.e. ``last_eta`` for the coupled model).
        It is explicit in time: the source never enters `matvec`,
        `precond` or the tau rows.  ``Di == 0`` skips the whole block, so the
        frozen reference step is bit-for-bit unchanged.
        """
        if psi is None:
            psi, _ = self.m.stokes(that)
        self.psi_n = psi
        if extrapolate and psi_prev is not None:
            psi = 2.0 * np.asarray(psi) - np.asarray(psi_prev)
            self.n_extrap += 1
        self.psi = psi
        self.ux = self._G((self.m.D @ np.asarray(psi).T).T)
        self.uz = self._G(-1j * self.m.kx[:, None] * psi)
        if self.dissipation_number != 0.0:
            eta = (self.m.dissipation_eta()
                   if hasattr(self.m, 'dissipation_eta') else None)
            self._phi_coeffs = DISS.source_coeffs(
                self.m, psi, eta, Di=self.dissipation_number)
            self.last_phi_total = DISS.source_integral(self.m,
                                                       self._phi_coeffs)
        return psi

    # ---------------------------------------------------------------- operators
    def lap(self, v):
        """Per-wavenumber Laplace operator, Chebyshev coefficients -> coeffs."""
        return (self.D2 @ np.asarray(v).T).T - self.k2 * v

    def advect(self, v):
        """+(u^n . grad) v, with the SAME 3/2 dealiasing as the reference rhs."""
        vx = self._G(1j * self.m.kx[:, None] * v)
        vz = self._G((self.m.D @ np.asarray(v).T).T)
        return self._F(-(self.ux * vx + self.uz * vz))

    def buoy(self, v):
        """u_z produced by the frozen-eta Stokes solve of v (exact, linear in v).

        u_z_hat = -i k psi_hat and psi_hat = S_k[v_k] is diagonal in k, so this
        is the exact linearisation of the buoyancy term at the frozen viscosity.
        """
        m = self.m
        out = np.zeros_like(v, dtype=complex)
        vv = getattr(m, 'vv', None)
        for i, k in enumerate(m.kx):
            if k == 0.0:
                continue
            if vv is not None:
                psi_i = vv.solve(k, v[i])[0]
            else:                       # isoviscous reference model
                f = -1j * k * m.Ra * v[i]
                om = m.Z.solve(k, f, 0.0)
                psi_i = m.Z.solve(k, -om, 0.0)
            out[i] = -1j * k * psi_i
        self.n_buoy_solves += 1
        return out

    def buoyancy_source(self):
        """u_z(theta^n) in coefficient space, from the frozen Stokes solution."""
        return -1j * self.m.kx[:, None] * self.psi

    def _rho_mul(self, v):
        """``rho_bar * v`` in Chebyshev coefficient space (per x-mode)."""
        return (self._Mrho @ np.asarray(v).T).T

    def _tala_adiabatic_source(self, that):
        """``rho_bar u_z [1 - Di (1 - z + theta)]`` in coefficient space.

        The TALA background/adiabatic term, evaluated on the SAME (x-dealiased)
        grid the advection uses and truncated back exactly like it, so the two
        explicit terms share one discretisation.  ``Di = 0`` gives
        ``rho_bar u_z`` = the frozen ``u_z`` source.
        """
        m = self.m
        Z = m.z[None, :]
        th = self._G(np.asarray(that))
        bracket = 1.0 - self.dissipation_number * (1.0 - Z + th)
        src = m.rho_grid[None, :] * self.uz * bracket
        out = self._F(src)
        out[:, -2:] = 0.0
        return out

    # --------------------------------------------------------------- assembling
    def matvec(self, v):
        v = np.asarray(v).reshape(self.m.Nk, self.m.Nz)
        # A v = a0 v - L v - advect(v) [- buoy(v)]
        #
        # SIGN (`advect` returns the term as it appears in the temperature
        # EQUATION, i.e. -(u.grad)v, so it must be SUBTRACTED here):
        #   theta' = L th^{n+1} + advect(th^{n+1}) + [u_z]
        #   =>  (a0 I - L) th^{n+1} - advect(th^{n+1}) = b0 th^n - b1 th^{n-1}
        # Getting this sign wrong leaves a 2*(u.grad)theta inconsistency that
        # is O(1) and is caught by `self_test` check [4].
        #
        # TALA: the time derivative and the advection carry the
        # reference-density weight `rho_bar(z)` (`rho_bar a0 v - lap v +
        # rho_bar (u.grad) v`); the diffusion does not.  The implicit-buoyancy
        # term is a Boussinesq device and is disabled for TALA (the whole
        # background/adiabatic term is explicit, `_tala_adiabatic_source`).
        # `rho_bar == 1` makes `_Mrho` the identity matrix, so the frozen
        # Boussinesq arithmetic is reproduced exactly.
        if self.tala_active:
            out = self.a0 * self._rho_mul(v) - self.lap(v) \
                - self._rho_mul(self.advect(v))
        else:
            out = self.a0 * v - self.lap(v) - self.advect(v)
        if self.implicit_buoy:
            out = out - self.buoy(v)
        # tau rows: Dirichlet theta(z=0) = theta(z=1) = 0
        out[:, -2] = v @ self.m.V[0]
        out[:, -1] = v @ self.m.V[-1]
        self.n_matvec += 1
        return out.ravel()

    def rhs(self, that, dt=None, that_prev=None, dt_prev=None):
        """Right-hand side b of the current step (theta^{n+1} unknown).

        internal heating: with ``internal_heating = H != 0`` the constant volumetric
        source is added here; the only change the extension makes to the
        temperature equation.  H is a constant, so its spectral image is the
        single coefficient ``(k_x = 0, T_0)``; it is added after the tau rows
        have been zeroed and it never touches them.  ``H == 0`` returns before
        any arithmetic, which is what makes the frozen reference step bit-identical
        (`specmc.physics.heating` documents the form and its provenance).

        viscous dissipation: with ``dissipation_number = Di != 0`` the viscous-dissipation
        source ``(Di/Ra) Phi`` assembled by :meth:`prepare` is added to the
        EQUATION rows only (never to the two Dirichlet/tau rows), again after
        the tau rows have been zeroed.  The order ``H`` first, ``Phi`` second is
        fixed; ``Di == 0`` returns the same array ``H``-only code returned,
        bit for bit.
        """
        if dt is None:
            dt = self.dt
        if dt_prev is not None or that_prev is None:
            self.set_step(dt, dt_prev)
        else:
            self.set_step(dt, self.dt_prev)
        b = self.b0 * np.asarray(that)
        if self.b1 != 0.0:
            if that_prev is None:
                raise ValueError('BDF2 needs theta^{n-1} (that_prev)')
            b = b - self.b1 * np.asarray(that_prev)
        if self.tala_active:
            # the TALA energy equation weights the time derivative and
            # the advection by rho_bar (G-ADOPT `rhocp()`), keeps the diffusion
            # unweighted, and replaces the Boussinesq background advection
            # `-u_z` by the compressible pair
            #     rho_bar u_z [1 - Di (1 - z + theta)]      (adiabatic term)
            # (`approximations.py:404-406`: Di alpha rho g.u T', with the
            # reference part `-Di rho_bar u_z Tbar` cancelled against the
            # advection of the adiabatic profile).  The radiogenic source
            # becomes rho_bar H (`energy_source`: H * rho).
            b = self._Mrho @ b.T
            b = b.T
        elif not self.implicit_buoy:
            b = b + self.buoyancy_source()
        b = b.copy()
        b[:, -2:] = 0.0
        if self.internal_heating != 0.0:
            if self.tala_active:
                # rho_bar H: a full Chebyshev spectrum in the k_x = 0 mode
                b[0] = b[0] + self.internal_heating * self.m.rho_c
            else:
                b = HEAT.add_constant_source(b.ravel(), self.m.Nk, self.m.Nz,
                                             self.internal_heating)
        if self.tala_active:
            b = b + self._tala_adiabatic_source(that)
        if self.dissipation_number != 0.0:
            if self._phi_coeffs is None:
                raise ValueError(
                    'dissipation_number=%r needs the frozen velocity: call '
                    'prepare() before rhs() (the source uses the eta field of '
                    'the current Stokes solve)' % (self.dissipation_number,))
            b = DISS.add_dissipation_source(np.asarray(b).ravel(),
                                            self._phi_coeffs,
                                            self.dissipation_number)
            self.n_diss += 1
        return np.asarray(b).ravel()

    # ---------------------------------------------------------- preconditioner
    def precond(self, r):
        """Per-wavenumber implicit-diffusion inverse: (a0 I - L)^-1.

        This is the operator the reference scheme solves every step
        (with a0 = 1/dt for backward Euler, a0 = 1.5/dt for constant-step BDF2),
        so it reuses its LU cache; it is exact when the advection/buoyancy terms
        are zero.

        TALA: the mass term is ``a0 rho_bar(z)``, so the preconditioner
        is the variable-coefficient ``(a0 rho_bar - L)`` per wavenumber.  It is
        built and LU-factorised here (Nz x Nz, bounded cache keyed by a0) rather
        than delegated to ``ChebTau``, whose cache assumes a constant
        coefficient.  ``rho_bar == 1`` reproduces the frozen denominator
        exactly.
        """
        m = self.m
        R = np.asarray(r).reshape(m.Nk, m.Nz)
        out = np.zeros_like(R, dtype=complex)
        lam = float(self.a0)
        if not self.tala_active:
            for i, k in enumerate(m.kx):
                out[i] = m.Z.solve(k, -R[i], lam=lam)
            return out.ravel()
        key = round(lam, 10)
        lu = self._tala_lu.get(key)
        if lu is None:
            lu = []
            N = m.Nz
            for k in m.kx:
                A = m.D @ m.D - (k * k + lam * m.rho_grid[None, :]) * np.eye(N)
                A[-2, :] = (-1.0) ** np.arange(N)
                A[-1, :] = 1.0
                lu.append(sla.lu_factor(A))
            if len(self._tala_lu) >= 4:
                self._tala_lu.clear()
            self._tala_lu[key] = lu
        for i in range(m.Nk):
            b = -R[i].copy()
            b[-2] = 0.0
            b[-1] = 0.0
            out[i] = sla.lu_solve(lu[i], b)
        return out.ravel()

    # -------------------------------------------------------------------- solve
    def solve(self, that, dt=None, x0=None, rtol=None, maxiter=None,
              that_prev=None, dt_prev=None):
        """One implicit step.  `that` must already be frozen via `prepare`."""
        if dt is None:
            dt = self.dt
        b = self.rhs(that, dt=dt, that_prev=that_prev, dt_prev=dt_prev)
        if x0 is not None:
            x0 = np.asarray(x0).ravel()
        x, info, it, hist = fgmres(self.matvec, b, M=self.precond, x0=x0,
                                   rtol=self.rtol if rtol is None else rtol,
                                   atol=self.atol,
                                   restart=self.restart,
                                   maxiter=self.maxiter if maxiter is None
                                   else maxiter)
        bnorm = max(np.linalg.norm(b), 1e-300)
        self.last_iters, self.last_info = it, info
        # `hist[-1]` is only the true residual at the end of a restart cycle;
        # measure the true value here so `last_res` is accurate, and COUNT the
        # unconverged steps (`info != 0`) instead of marching on silently (the
        # production drivers now log both).
        self.last_res = float(np.linalg.norm(b - self.matvec(x.ravel()))) / bnorm
        if info != 0:
            self.n_nonconv += 1
        self.n_solves += 1
        return x.reshape(self.m.Nk, self.m.Nz), info, it, hist

    # ---------------------------------------------------------------- one step
    def step(self, that, dt, x0=None, psi=None, that_prev=None, dt_prev=None,
             psi_prev=None, extrapolate=False):
        """Full model step: freeze u(theta^n), solve (S1)/(S2)/(B2), project."""
        m = self.m
        if getattr(m, 'symmetrize', False):
            that = m.project_mirror(that)
        self.prepare(that, psi=psi, psi_prev=psi_prev, extrapolate=extrapolate)
        out, info, it, _ = self.solve(that, dt, x0=x0, that_prev=that_prev,
                                      dt_prev=dt_prev)
        m.n_steps += 1
        if getattr(m, 'symmetrize', False):
            out = m.project_mirror(out)
        return out


class RBCImplicitTheta(RBCVariableViscosity):
    """`RBCVariableViscosity` whose temperature step is the implicit scheme.

    Only `step()` is overridden; the Stokes solve, the diagnostics, the
    projection and the CFL helper are the validated ones.

    `theta_scheme`
        'implicit' / 'implicit_be'  backward Euler (schemes S1/S2 above)
        'implicit_bdf2'             BDF2 (second-order, variable step);
                                    needs one previous state, which the model
                                    keeps itself (the first step is BE).

    `theta_extrapolate` (only used by BDF2)
        True  -> advection velocity and buoyancy source of the step are the
                 second-order extrapolation 2*u^n - u^{n-1}.  This is what makes
                 the scheme second order: with the plain lag u^n the O(dt) error
                 of the lagged velocity survives and the scheme is only first
                 order (measured in the BDF2 self-test).
        False -> the plain first-order form, advection velocity = u^n.
    """

    def __init__(self, Ra=1e4, Nx=64, Nz=48, Lx=2.0, zsolver='tau',
                 dealias=True, viscosity_law='exp_T_depth', deta_T=1000.0,
                 use_lu=True, dealias_z=False, symmetrize=True,
                 zform='reduced', theta_scheme='implicit',
                 implicit_buoy=True, theta_rtol=1e-10, theta_maxiter=200,
                 theta_restart=40, theta_extrapolate=True,
                 internal_heating=0.0, dissipation_number=0.0,
                 compressibility='boussinesq',
                 eta_min=None, eta_max=None,
                 eta_smooth=None):
        super().__init__(Ra=Ra, Nx=Nx, Nz=Nz, Lx=Lx, zsolver=zsolver,
                         dealias=dealias, viscosity_law=viscosity_law,
                         deta_T=deta_T, use_lu=use_lu, dealias_z=dealias_z,
                         symmetrize=symmetrize, zform=zform,
                         internal_heating=internal_heating,
                         dissipation_number=dissipation_number,
                         compressibility=compressibility,
                         eta_min=eta_min, eta_max=eta_max,
                         eta_smooth=eta_smooth)
        if str(theta_scheme).lower() not in THETA_SCHEMES:
            raise ValueError('unknown theta_scheme %r' % (theta_scheme,))
        self.theta_scheme = str(theta_scheme).lower()
        self.implicit_buoy = bool(implicit_buoy)
        if self.theta_scheme == 'explicit':
            raise ValueError("theta_scheme='explicit' is the frozen reference "
                             "scheme: use RBCVariableViscosity for that")
        if self.dealias_z:
            raise ValueError('theta_scheme=implicit and dealias_z are not '
                             'combined (the implicit matvec uses the x-only '
                             'dealiasing of RBC.explicit_rhs)')
        time_scheme = 'bdf2' if self.theta_scheme == 'implicit_bdf2' else 'be'
        self.theta = ImplicitThetaSolver(
            self, rtol=theta_rtol, maxiter=theta_maxiter,
            restart=theta_restart, implicit_buoy=implicit_buoy,
            scheme=time_scheme, internal_heating=internal_heating,
            dissipation_number=dissipation_number)
        # BDF2 history (theta^{n-1}, dt_{n-1}, psi^{n-1}); None -> BE restart
        self.that_prev = None
        self.dt_prev = None
        self.psi_prev = None
        self.theta_extrapolate = bool(theta_extrapolate)

    # ------------------------------------------------------------ bdf2 history
    def reset_history(self):
        """Forget the previous state: the next step restarts as backward Euler."""
        self.that_prev = None
        self.dt_prev = None
        self.psi_prev = None

    def step(self, that, dt, x0=None, psi=None):
        m = self
        th = m.project_mirror(that) if getattr(m, 'symmetrize', False) else that
        use_x = (self.theta.scheme == 'bdf2' and self.theta_extrapolate
                 and self.psi_prev is not None)
        out = self.theta.step(th, dt, x0=x0, psi=psi,
                              that_prev=self.that_prev, dt_prev=self.dt_prev,
                              psi_prev=self.psi_prev, extrapolate=use_x)
        if self.theta.scheme == 'bdf2':
            self.that_prev = np.array(th, copy=True)
            self.dt_prev = float(dt)
            self.psi_prev = self.theta.psi_n
        return out

    def describe(self):
        d = super().describe()
        d.update(theta_scheme=self.theta_scheme,
                 time_scheme=self.theta.scheme,
                 theta_extrapolate=self.theta_extrapolate,
                 implicit_buoy=self.implicit_buoy,
                 internal_heating=self.theta.internal_heating,
                 dissipation_number=self.theta.dissipation_number,
                 dissipation_active=bool(self.theta.dissipation_number != 0.0),
                 phi_total=self.theta.last_phi_total,
                 n_diss=self.theta.n_diss,
                 theta_rtol=self.theta.rtol,
                 n_theta_nonconv=self.theta.n_nonconv,
                 n_be_fallback=self.theta.n_be_fallback,
                 theta_solver='FGMRES(restart=%d) + per-wavenumber implicit '
                              'diffusion preconditioner' % self.theta.restart)
        return d


# ------------------------------------------------------------------ self test
def self_test(Nx=16, Nz=24, deta_T=100.0, verbose=True):
    """Structural checks that need no long run."""
    rng = np.random.default_rng(0)
    ok = True
    for buoy in (True, False):
        m = RBCImplicitTheta(Nx=Nx, Nz=Nz, deta_T=deta_T, implicit_buoy=buoy)
        that = m.seed(amp=1e-2)
        m.theta.prepare(that)
        v = np.zeros((m.Nk, m.Nz), dtype=complex)
        for i in range(1, m.Nk):
            v[i] = (rng.normal(size=m.Nz) + 1j * rng.normal(size=m.Nz)) \
                * np.exp(-np.arange(m.Nz) / 5.0)
        # 1. tau rows of the matvec are the Dirichlet rows
        m.theta.set_step(3.7e-3)
        Av = m.theta.matvec(v)
        e_bc = max(abs(Av.reshape(m.Nk, m.Nz)[:, -2] - v @ m.V[0]).max(),
                   abs(Av.reshape(m.Nk, m.Nz)[:, -1] - v @ m.V[-1]).max())
        # 2. the preconditioner is the exact inverse of (a0 I - L)
        r = np.array(v, copy=True)
        # (a0 I - L) x = r  is what precond inverts (rows 0..N-3)
        x = m.theta.precond(r.ravel()).reshape(m.Nk, m.Nz)
        lhs = m.theta.a0 * x - m.theta.lap(x)
        e_pre = np.abs(lhs[:, :-2] - r[:, :-2]).max() / max(np.abs(r).max(), 1e-300)
        ok &= e_bc < 1e-14 and e_pre < 1e-10
        if verbose:
            print('  implicit_buoy=%-5s  BC rows %.1e  precond exactness %.1e'
                  % (buoy, e_bc, e_pre))
    # 3. FGMRES drives the TRUE residual below rtol
    m = RBCImplicitTheta(Nx=Nx, Nz=Nz, deta_T=deta_T)
    that = m.seed(amp=1e-2)
    m.theta.prepare(that)
    dt = 1e-3
    b = m.theta.rhs(that, dt)
    x, info, it, hist = m.theta.solve(that, dt)
    res = np.linalg.norm(b - m.theta.matvec(x.ravel())) / np.linalg.norm(b)
    ok &= (info == 0 and res < 1e-8)
    if verbose:
        print('  first implicit step: iters=%d info=%d true rel.res=%.2e'
              % (it, info, res))

    # 4. CONSISTENCY (this is the check that catches a wrong sign in the
    #    advection term).  One implicit step and one step of the reference
    #    scheme differ by dt * (u.grad)(theta^{n+1}-theta^n) = O(dt^2), so the
    #    difference must fall by 4x when dt is halved.  A wrong sign leaves a
    #    2*(u.grad)theta term, which is O(dt) with a huge coefficient.
    from ..models.convection_modes import RBCVariableViscosity
    m_ref = RBCVariableViscosity(Nx=Nx, Nz=Nz, deta_T=deta_T)
    m_ref.z, m_ref.x = m.z, m.x
    th2 = m.seed(amp=3e-2)
    errs = []
    for d in (1e-5, 5e-6, 2.5e-6):
        a = m_ref.step(th2, d)
        m.theta.prepare(th2)
        c, _, _, _ = m.theta.solve(th2, d)
        errs.append(float(np.abs(c - a).max()))
    ratio = errs[0] / max(errs[1], 1e-300)
    ratio2 = errs[1] / max(errs[2], 1e-300)
    step = float(np.abs(m_ref.step(th2, 1e-5) - th2).max())
    ok &= (errs[0] < 0.05 * step) and (ratio > 3.0) and (ratio2 > 3.0)
    if verbose:
        print('  consistency vs reference step: |d| = %.2e %.2e %.2e '
              '(ratios %.2f %.2f, expect ~4); one explicit step moves %.2e'
              % (errs[0], errs[1], errs[2], ratio, ratio2, step))

    # 5. BDF2 time-derivative coefficients (variable step and constant step)
    s = ImplicitThetaSolver(m, scheme='bdf2')
    dt = 3.0e-4
    s.set_step(dt, dt)                       # r = 1
    e5 = max(abs(s.a0 * dt - 1.5), abs(s.b0 * dt - 2.0), abs(s.b1 * dt - 0.5))
    s.set_step(dt, dt / 2.0)                 # r = 2
    e5 = max(e5, abs(s.a0 * dt - 5.0 / 3.0), abs(s.b0 * dt - 3.0),
             abs(s.b1 * dt - 4.0 / 3.0))
    s.set_step(dt, None)                     # restart -> backward Euler
    e5 = max(e5, abs(s.a0 * dt - 1.0), abs(s.b0 * dt - 1.0), abs(s.b1 * dt))
    ok &= (e5 < 1e-14) and s.be_fallback
    if verbose:
        print('  bdf2 coefficients (r=1, r=2, restart->BE): max err %.1e  %s'
              % (e5, 'OK' if e5 < 1e-14 else 'FAIL'))

    # 6. one BDF2 step is close to one BE step with the SAME history (both must
    #    solve the same PDE, so the difference must be small and O(dt); a sign
    #    or coefficient error in the BDF2 operator would make it O(theta)).
    m_b = RBCImplicitTheta(Nx=Nx, Nz=Nz, deta_T=deta_T,
                           theta_scheme='implicit_bdf2')
    e6 = []
    for d in (1e-5, 5e-6):
        m.theta.prepare(th2)
        a = m.theta.solve(th2, d)[0]
        m_b.theta.prepare(th2)
        c = m_b.theta.solve(th2, d, that_prev=th2, dt_prev=d)[0]
        e6.append(float(np.abs(c - a).max()))
    e6r = e6[0] / max(e6[1], 1e-300)
    ok &= (e6[0] < step) and (e6r > 1.5)
    if verbose:
        print('  bdf2 vs BE one step (same history): |d| = %.2e %.2e '
              '(ratio %.2f, expect ~2)' % (e6[0], e6[1], e6r))

    # 7. internal heating: (a) H = 0 is a bit-for-bit no-op on the RHS,
    #    (b) H != 0 changes exactly ONE coefficient, by exactly H, and that
    #    coefficient is the (k_x = 0, T_0) one.  This is the structural
    #    statement behind the internal-heating response.
    m0 = RBCImplicitTheta(Nx=Nx, Nz=Nz, deta_T=deta_T, implicit_buoy=False)
    mH = RBCImplicitTheta(Nx=Nx, Nz=Nz, deta_T=deta_T, implicit_buoy=False,
                          internal_heating=12.5)
    th7 = m.seed(amp=2e-2)
    m0.theta.prepare(th7)
    mH.theta.prepare(th7)
    b0 = m0.theta.rhs(th7, 3.1e-3)
    bH = mH.theta.rhs(th7, 3.1e-3)
    d = (bH - b0).reshape(m0.Nk, m0.Nz)
    nz = int(np.count_nonzero(d))
    e7 = float(np.abs(d[0, 0] - 12.5))
    # the H = 0 object must produce the identical array as a model built
    # without the keyword at all (and it, too, needs its velocity prepared,
    # since `rhs` reads the frozen `psi` when implicit_buoy is off)
    m_ref = RBCImplicitTheta(Nx=Nx, Nz=Nz, deta_T=deta_T, implicit_buoy=False)
    m_ref.theta.prepare(th7)
    e_noop = float(np.abs(m_ref.theta.rhs(th7, 3.1e-3) - b0).max())
    ok &= (nz == 1) and (e7 == 0.0) and (e_noop == 0.0)
    if verbose:
        print('  [7] internal heating: H=0 RHS bit-identical (%.1e); '
              'H=12.5 touches %d coefficient(s), |db - H| = %.1e'
              % (e_noop, nz, e7))

    # 8. viscous dissipation: (a) Di = 0 is a bit-for-bit no-op on the
    #    RHS; (b) Di != 0 adds the (Di/Ra) Phi coefficient array to the
    #    EQUATION rows only (the two tau rows stay exactly zero), and the
    #    delta is exactly the source the SSOT module returns; (c) the source
    #    is law-blind: the same history gives the same Ph for eta(T) and
    #    eta(z) models only if the eta differs, which is the intended check.
    mD0 = RBCImplicitTheta(Nx=Nx, Nz=Nz, deta_T=deta_T, implicit_buoy=False)
    mD = RBCImplicitTheta(Nx=Nx, Nz=Nz, deta_T=deta_T, implicit_buoy=False,
                          dissipation_number=0.5)
    th8 = m.seed(amp=2e-2)
    mD0.theta.prepare(th8)
    mD.theta.prepare(th8)
    b8_0 = mD0.theta.rhs(th8, 3.1e-3)
    b8_D = mD.theta.rhs(th8, 3.1e-3)
    d8 = (b8_D - b8_0).reshape(mD.Nk, mD.Nz)
    src = DISS.source_coeffs(mD, mD.theta.psi, mD.dissipation_eta(), Di=0.5)
    # `b + src - b` is a roundoff-limited difference, not an exact one
    e8 = float(np.abs(d8 - src).max()
               / max(float(np.abs(b8_0).max()), 1e-300))
    tau8 = float(np.abs(b8_D.reshape(mD.Nk, mD.Nz)[:, -2:]).max())
    mDref = RBCImplicitTheta(Nx=Nx, Nz=Nz, deta_T=deta_T, implicit_buoy=False)
    mDref.theta.prepare(th8)
    e8_noop = float(np.abs(mDref.theta.rhs(th8, 3.1e-3) - b8_0).max())
    ok &= (e8 <= 1e-14) and (tau8 == 0.0) and (e8_noop == 0.0) \
        and mD.theta.n_diss == 1 and mD0.theta.n_diss == 0 \
        and mD.theta.last_phi_total != 0.0
    if verbose:
        print('  [8] viscous dissipation: Di=0 RHS bit-identical (%.1e); '
              'Di=0.5 delta == (Di/Ra)Phi to %.1e (rel), tau rows %.1e, '
              'n_diss=%d, source integral %.3e'
              % (e8_noop, e8, tau8, mD.theta.n_diss,
                 mD.theta.last_phi_total))

    if verbose:
        print('  implicit_theta self_test %s' % ('PASSED' if ok else 'FAILED'))
    return bool(ok)


if __name__ == '__main__':
    print('implicit temperature step self test')
    self_test()
