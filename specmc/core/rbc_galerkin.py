"""Corrected 2D infinite-Prandtl Rayleigh-Benard solver (spectral, direct).

Geometry and benchmark
----------------------
1 x 1 box, free-slip everywhere, isothermal top/bottom, insulating side walls
(Blankenbach et al. 1989, case 1a:  Ra = 1e4  ->  Nu = 4.884, Vrms = 42.8649).

The box is discretised as a PERIODIC grid of length Lx = 2 with Nx points.  On
that grid kx = 2*pi*m/2 = pi*m; an initial condition that is mirror-symmetric
about x = 0 and x = 1 stays exactly mirror-symmetric, and that symmetric
subspace *is* the free-slip / insulating 1x1 box (the second half is the mirror
image).  `mirror_error()` verifies this at run time.

Governing equations (unit viscosity, unit thermal diffusivity, Lz = 1)

    T                 = (1 - z) + theta            T = 1 at z = 0 (hot bottom)
    lap(psi) + omega  = 0                          omega = -lap(psi)
    lap(omega)        = -Ra d(theta)/dx            (curl of the Stokes equation)
    d(theta)/dt + u.grad(theta) - u_z = lap(theta) + H
    u                 = (d psi/dz, -d psi/dx)

`H` is the constant internal heating rate (`specmc.physics.heating`);
``internal_heating = 0`` (the default, and the frozen reference value) leaves the
equation, and every floating-point operation of it, exactly as it was.

Free-slip  :  psi = 0  and  psi_zz = 0  at z = 0,1
              psi = 0  and  psi_xx = 0  at x = 0,1   (implied by the mirror
                                                        symmetry of the grid)
Isothermal :  theta = 0 at z = 0,1
Insulating :  d(theta)/dx = 0 at x = 0,1

The four free-slip conditions are enforced EXACTLY by the two scalar Dirichlet
problems (both derived from the same operator D^2 - k^2):

    (D^2 - k^2) omega = -Ra i k theta_hat ,   omega(0) = omega(1) = 0    (1)
    (D^2 - k^2) psi   = -omega             ,   psi(0)   = psi(1)   = 0    (2)

because  omega = -lap(psi) = -(psi_zz - k^2 psi)  vanishes on the wall whenever
psi = psi_zz = 0.  Eliminating omega gives the physical biharmonic
lap(lap(psi)) = Ra d(theta)/dx.

Sign convention
-----------------------------------------------------
   theta = cos(kx x) sin(kz z),  gamma = kx^2 + kz^2
   psi   = -Ra kx/gamma^2 * sin(kx x) sin(kz z)
   omega = -gamma psi
   u_z   = -d psi/dx = +Ra kx^2/gamma^2 * theta      <- hot fluid RISES

The temperature source is +u_z (advection of the conductive profile
T = 1 - z gives u.grad(T) = -u_z for rising fluid), not -u_z.

z-discretisation: Chebyshev-tau (default) or a Dirichlet Galerkin / collocation
basis phi_j = T_j - T_{j+2} (`zsolver='galerkin'`).  Both solve the same
discrete operator; they agree to machine precision (package self-tests).
The operator is time independent, so every wave number is factorised ONCE.
"""
import numpy as np
import scipy.linalg as sla
from numpy.polynomial import chebyshev as _C


# --------------------------------------------------------------------- bases
def cgl(N):
    """Chebyshev-Gauss-Lobatto nodes on [0,1], ascending, endpoints included."""
    return 0.5 * (1.0 - np.cos(np.pi * np.arange(N) / (N - 1)))


def cheb_D(N):
    """Coefficient-space d/dz on [0,1] for u = sum_k c_k T_k(2z-1)."""
    D = np.zeros((N, N))
    for k in range(1, N):
        c = np.zeros(N)
        c[k] = 1.0
        d = 2.0 * _C.chebder(c)
        D[:len(d), k] = d
    return D


def cheb_V(z, N):
    """V[j,k] = T_k(2 z_j - 1)."""
    return _C.chebvander(2.0 * np.asarray(z, float) - 1.0, N - 1)


# ----------------------------------------------------------------- z solvers
class _CachedSolveMixin:
    """Shared bookkeeping for the two z solvers.

    Two caches:
      * `_lu0`, LU factors of the lam = 0 operator.  Exactly one entry per
        horizontal wavenumber, built once and reused forever.  This is the
        Stokes operator, so every later Stokes solve is one back substitution.
      * `_lut`, LU factors of the shifted (implicit diffusion) operator,
        keyed by lam = 1/dt.  dt varies during the transient, so this cache is
        BOUNDED (`max_lut`).  Leaving it unbounded is a real memory bug: 33
        wavenumbers x a fresh key per timestep x Nz^2 float64 grows without
        limit and eventually kills the process.

    With `use_lu=False` the fallback is `np.linalg.solve`, which re-factorises
    on every call; that is what the earlier timings measured.
    """

    max_lut = 4          # max number of distinct lam = 1/dt values kept

    def _init_cache(self, use_lu=True):
        self._lu0, self._lut, self._cache = {}, {}, {}
        self.n_fact, self.use_lu = 0, bool(use_lu)

    def matrix(self, k, lam=0.0):
        """Dense operator (no factorisation).  lam = 0 is cached; lam != 0 not."""
        key = (round(float(k), 10), round(float(lam), 10))
        if lam == 0.0:
            A = self._cache.get(key)
            if A is None:
                A = self._assemble(k, lam)
                self._cache[key] = A
            return A
        return self._assemble(k, lam)

    def _factors_for(self, k, lam):
        if not self.use_lu:
            return None
        key = round(float(k), 10)
        if lam == 0.0:
            lu = self._lu0.get(key)
            if lu is None:
                lu = sla.lu_factor(self.matrix(k, 0.0))
                self._lu0[key] = lu
                self.n_fact += 1
            return lu
        # nest by lam: ONE dt value must hold all Nk wavenumbers, so bound the
        # number of distinct dt values, never the number of (k, lam) pairs.
        lkey = round(float(lam), 10)
        sub = self._lut.get(lkey)
        if sub is None:
            if len(self._lut) >= self.max_lut:
                self._lut.clear()
            sub = {}
            self._lut[lkey] = sub
        lu = sub.get(key)
        if lu is None:
            lu = sla.lu_factor(self._assemble(k, lam))
            sub[key] = lu
            self.n_fact += 1
        return lu


class ChebTau(_CachedSolveMixin):
    """Chebyshev-tau direct solver for  u'' - (k^2 + lam) u = rhs ,  u(0)=u(1)=0.

    Rows 0..N-3 are the differential equation, rows N-2, N-1 are the boundary
    conditions (the two tau parameters are eliminated by setting those two
    right-hand sides to zero).
    """

    def __init__(self, N, use_lu=True):
        self.N = N
        self.z = cgl(N)
        self.V = cheb_V(self.z, N)                     # values = V @ coeffs
        self.Vi = np.linalg.inv(self.V)                # coeffs = Vi @ values
        self.D = cheb_D(N)
        self.D2 = self.D @ self.D
        self._init_cache(use_lu)

    def _assemble(self, k, lam):
        N = self.N
        A = self.D2 - (k * k + lam) * np.eye(N)
        A[-2, :] = (-1.0) ** np.arange(N)              # u(0) = 0
        A[-1, :] = 1.0                                 # u(1) = 0
        return A

    def solve(self, k, rhs, lam=0.0):
        """rhs and return value are Chebyshev COEFFICIENT vectors."""
        b = np.array(rhs, dtype=complex, copy=True)
        b[-2] = 0.0
        b[-1] = 0.0
        lu = self._factors_for(k, lam)
        if lu is None:
            A = self.matrix(k, lam)
            if np.iscomplexobj(rhs):
                return (np.linalg.solve(A, b.real)
                        + 1j * np.linalg.solve(A, b.imag))
            return np.linalg.solve(A, b.real)
        if np.iscomplexobj(rhs):
            return sla.lu_solve(lu, b.real) + 1j * sla.lu_solve(lu, b.imag)
        return sla.lu_solve(lu, b.real)


class ChebGalerkin(_CachedSolveMixin):
    """Dirichlet Galerkin / collocation solver on the BC-adapted basis

        phi_j(z) = T_j(xi) - T_{j+2}(xi),   xi = 2z-1,   j = 0 .. N-3

    phi_j(0) = phi_j(1) = 0 for every j, so the homogeneous Dirichlet
    conditions are built into the trial space.  The N-2 expansion coefficients
    are fixed by collocating  u'' - (k^2+lam) u = rhs  at the N-2 interior
    CGL nodes.  No tau rows are used.
    """

    def __init__(self, N, use_lu=True):
        self.N = N
        self.M = N - 2
        self.z = cgl(N)
        self.V = cheb_V(self.z, N)
        self.Vi = np.linalg.inv(self.V)
        self.D = cheb_D(N)
        self.zi = self.z[1:-1]
        self.Vint = cheb_V(self.zi, N)                 # full series at interior nodes
        V1 = self.Vint @ self.D
        V2 = V1 @ self.D
        M = self.M
        self.B = self.Vint[:, :M] - self.Vint[:, 2:M + 2]
        self.B1 = V1[:, :M] - V1[:, 2:M + 2]
        self.B2 = V2[:, :M] - V2[:, 2:M + 2]
        self._init_cache(use_lu)

    def _assemble(self, k, lam):
        return self.B2 - (k * k + lam) * self.B

    def expand(self, a):
        c = np.zeros(self.N, dtype=complex if np.iscomplexobj(a) else float)
        c[:self.M] += a
        c[2:self.M + 2] -= a
        return c

    def solve(self, k, rhs, lam=0.0):
        r = np.asarray(rhs)
        b = self.Vint @ r
        lu = self._factors_for(k, lam)
        if lu is None:
            A = self.matrix(k, lam)
            if np.iscomplexobj(r):
                a = np.linalg.solve(A, b.real) + 1j * np.linalg.solve(A, b.imag)
            else:
                a = np.linalg.solve(A, b)
            return self.expand(a)
        if np.iscomplexobj(r):
            a = sla.lu_solve(lu, b.real) + 1j * sla.lu_solve(lu, b.imag)
        else:
            a = sla.lu_solve(lu, b.real)
        return self.expand(a)


#: accepted values of the TALA `compressibility` switch.  Only these two: ALA
#: is explicitly out of scope.
COMPRESSIBILITY_MODES = ("boussinesq", "tala")


def _normalize_compressibility(c):
    """-> the canonical `compressibility` string, or raise."""
    s = str(c).lower()
    if s not in COMPRESSIBILITY_MODES:
        raise ValueError('unknown compressibility %r (use one of %s)'
                         % (c, ", ".join(COMPRESSIBILITY_MODES)))
    return s


# -------------------------------------------------------------------- model
class RBC:
    def __init__(self, Ra=1e4, Nx=64, Nz=48, Lx=2.0, zsolver='tau',
                 dealias=True, internal_heating=0.0, dissipation_number=0.0,
                 compressibility='boussinesq'):
        self.Ra, self.Nx, self.Nz, self.Lx = Ra, Nx, Nz, Lx
        #: constant internal heating rate H of the theta equation
        #: (`specmc.physics.heating`).  0.0 = the frozen reference temperature
        #: equation, bit for bit.
        self.internal_heating = float(internal_heating)
        #: viscous dissipation number Di (`specmc.physics.dissipation`);
        #: the source is ``(Di/Ra) Phi``.  0.0 (the frozen reference value) leaves the
        #: temperature equation, and every floating-point operation of it,
        #: exactly as it was; the source is not even evaluated.
        self.dissipation_number = float(dissipation_number)
        self.kx = 2 * np.pi * np.fft.rfftfreq(Nx, d=Lx / Nx)   # = pi*m
        self.Nk = len(self.kx)
        self.x = np.arange(Nx) * Lx / Nx
        self.Z = ChebTau(Nz) if zsolver == 'tau' else ChebGalerkin(Nz)
        self.zsolver = zsolver
        self.z, self.V, self.D, self.Vi = self.Z.z, self.Z.V, self.Z.D, self.Z.Vi
        self.dealias = bool(dealias)
        self.Np = 2 * Nx                       # padding for the 2/3 rule
        self.n_steps, self.n_stokes = 0, 0
        # Clenshaw-Curtis weights: sum_j wq_j f(z_j) = int_0^1 f dz.
        # The CGL nodes cluster near the walls, so the FLAT grid average is NOT
        # the volume average (it is wrong by ~30 % for <u_z theta> at Ra=1e4).
        gk = np.array([0.0 if (k % 2) else 1.0 / (1.0 - k * k)
                       for k in range(Nz)])
        gk[0] = 1.0
        self.wq = self.Vi.T @ gk
        # --- the TALA reference state --------------------------------
        # `compressibility='tala'` switches the Stokes operator, the velocity
        # reconstruction and the temperature equation to the anelastic forms of
        # `specmc.stokes.stokes_tala` / `specmc.physics.compressibility`.  The
        # ONE dissipation number `Di` of the run is
        # `dissipation_number`: it drives both the `(Di/Ra) Phi`
        # source and the reference density, and nothing here re-declares it.
        # `'boussinesq'` (the default) never evaluates any of these fields, so
        # the frozen reference path is untouched.
        self.compressibility = _normalize_compressibility(compressibility)
        self.tala_active = (self.compressibility == 'tala')
        if self.tala_active:
            from ..physics import compressibility as COMP
            self.tala_a = COMP.a_coeff(self.dissipation_number)
            self.rho_grid = COMP.rho_bar_grid(self.z, self.dissipation_number)
            self.rho_c = COMP.rho_bar_coeffs(self.Vi, self.z,
                                             self.dissipation_number)
            self.rho2_c = COMP.rho_bar_coeffs(
                self.Vi, self.z, 2.0 * self.dissipation_number)
            self.tala_record = COMP.record(self.dissipation_number)
        else:
            self.tala_a = 0.0
            self.rho_grid = None
            self.rho_c = None
            self.rho2_c = None
            self.tala_record = None
        for k in self.kx:                      # factorise the Stokes operator once
            self.Z._factors_for(k, 0.0)

    def vol_mean(self, g):
        """Volume average of a grid field (Nx,Nz) with correct z quadrature."""
        return float(np.sum(self.wq * np.asarray(g).real.mean(axis=0)))

    # ------------------------------------------------------------ transforms
    def grid(self, cf):
        """(Nk,Nz) Chebyshev coefficients per x-mode -> (Nx,Nz) grid values."""
        return np.fft.irfft((self.V @ np.asarray(cf).T).T * self.Nx,
                            n=self.Nx, axis=0)

    def coeffs(self, g):
        """(Nx,Nz) grid values -> (Nk,Nz) Chebyshev coefficients per x-mode."""
        A = np.fft.rfft(np.asarray(g), axis=0) / self.Nx
        return (self.Vi @ A.T).T

    def _synth_pad(self, cf):
        """Zero-padded synthesis on 2*Nx points (exact: input has Nx/2 modes)."""
        sp = np.zeros((self.Np // 2 + 1, cf.shape[1]), dtype=complex)
        sp[:self.Nk] = (self.V @ np.asarray(cf).T).T
        return np.fft.irfft(sp * self.Np, n=self.Np, axis=0)

    def _trunc_pad(self, g):
        return (np.fft.rfft(g, axis=0) / self.Np)[:self.Nk]

    # --------------------------------------------------------------- physics
    def stokes(self, that):
        """theta coefficients -> (psi, omega) coefficients (Chebyshev, per mode)."""
        psi = np.zeros_like(that, dtype=complex)
        om = np.zeros_like(that, dtype=complex)
        for i, k in enumerate(self.kx):
            if k == 0:
                continue                       # no horizontal buoyancy gradient
            f = -1j * k * self.Ra * that[i]    # (D^2-k^2) om = -Ra d(theta)/dx
            om[i] = self.Z.solve(k, f, 0.0)
            psi[i] = self.Z.solve(k, -om[i], 0.0)
        self.n_stokes += 1
        return psi, om

    def velocity(self, psi):
        """psi coefficients -> (ux, uz) on the (Nx,Nz) grid.

        TALA: under ``compressibility='tala'`` the velocity is recovered from
        the mass-flux streamfunction, ``u = curl^perp(Psi) / rho_bar`` (the
        whole point of the TALA formulation: ``div.(rho_bar u) = 0`` exactly).
        The division is the pointwise z-operation at the CGL nodes, so it is
        the exact inverse of the multiplication the mass flux uses.  With the
        default ``'boussinesq'`` the branch is not taken and the two lines are
        the frozen ones, bit for bit.
        """
        ux = self.grid((self.D @ np.asarray(psi).T).T)
        uz = self.grid(-1j * self.kx[:, None] * psi)
        if self.tala_active:
            r = self.rho_grid[None, :]
            ux = ux / r
            uz = uz / r
        return ux, uz

    # ------------------------------------------- viscous dissipation
    def dissipation_eta(self):
        """-> the eta field the Stokes operator consumed (None = eta == 1).

        `RBC` has no rheology at all, so the isoviscous solver returns None and
        `specmc.physics.dissipation` evaluates ``Phi`` with ``eta = 1``.
        `RBCVariableViscosity` returns its frozen ``eta_grid`` and `RBCCoupled`
        the field its last coupled solve consumed.
        """
        return None

    def dissipation_phi(self, psi, synth=None):
        """-> ``Phi`` on the (padded) grid for the frozen ``psi``.

        One definition for every model in the project: the algebra lives in
        `specmc.physics.dissipation`, and the synthesis is the production
        ``eta_pad_factor`` one (``synth`` lets the z-dealiased reference path
        pass its own ``_Gz``).  ``Di == 0`` never reaches this method.
        """
        from ..physics import dissipation as DISS
        return DISS.phi_padded(self, psi, self.dissipation_eta(), synth=synth)

    def explicit_rhs(self, that, psi):
        """theta_t = lap(theta) + N ;  N = -(u.grad)theta + u_z [+ H] [+ (Di/Ra)Phi].

        internal heating: with ``internal_heating = H != 0`` the constant volumetric
        source is added to the grid field.  A constant survives the rfft and
        the 2/3-rule truncation exactly (it is the k_x = 0, T_0 coefficient),
        so no dealiasing is involved.  ``H == 0`` returns the frozen reference
        expression unchanged, bit for bit.

        viscous dissipation: with ``dissipation_number = Di != 0`` the viscous-dissipation
        source ``(Di/Ra) Phi`` is added on the SAME padded grid, through the
        SAME pipeline.  ``Di == 0`` (the frozen reference value)
        skips the whole branch, so the explicit path is bit for bit unchanged.
        """
        def G(cf):
            return self._synth_pad(cf) if self.dealias else self.grid(cf)
        th = G(that)
        thx = G(1j * self.kx[:, None] * that)
        thz = G((self.D @ np.asarray(that).T).T)
        ux = G((self.D @ np.asarray(psi).T).T)
        uz = G(-1j * self.kx[:, None] * psi)
        N = -(ux * thx + uz * thz) + uz
        if self.internal_heating != 0.0:
            N = N + self.internal_heating
        if self.dissipation_number != 0.0:
            N = N + (self.dissipation_number / float(self.Ra)) \
                * self.dissipation_phi(psi)
        return N

    # ------------------------------------------------------------- time step
    def step(self, that, dt):
        psi, _ = self.stokes(that)
        N = self.explicit_rhs(that, psi)
        Nc = (self.Vi @ (self._trunc_pad(N) if self.dealias
                         else np.fft.rfft(N, axis=0) / self.Nx).T).T
        out = np.zeros_like(that, dtype=complex)
        for i, k in enumerate(self.kx):
            rhs = that[i] / dt + Nc[i]
            out[i] = self.Z.solve(k, -rhs, lam=1.0 / dt)
        self.n_steps += 1
        return out

    # ------------------------------------------------------------ diagnostics
    def diagnostics(self, that):
        psi, om = self.stokes(that)
        ux, uz = self.velocity(psi)
        th = np.real(self.grid(that))
        dth = self.D @ that[0]
        urms = float(np.sqrt(max(self.vol_mean(ux ** 2 + uz ** 2), 0.0)))
        return dict(
            Nu_top=float(1.0 - (self.V[-1] @ dth).real),
            Nu_bot=float(1.0 - (self.V[0] @ dth).real),
            # steady-state identity  Nu = 1 + <u_z theta>
            Nu_energy=float(1.0 + self.vol_mean(np.real(uz) * th)),
            urms=urms,
            umax=float(max(np.abs(ux).max(), np.abs(uz).max())),
            amp=float(np.abs(th).max()),
        )

    def mirror_error(self, that):
        """max |theta(x) - theta(Lx-x)| / max|theta| (must be ~1e-16).

        On x_j = 2j/Nx the reflection x -> Lx - x is the index map
        j -> (-j) mod Nx (NOT a plain reversal, which would reflect about
        x = 1 - 1/Nx)."""
        g = self.grid(that)
        idx = (-np.arange(self.Nx)) % self.Nx
        return float(np.abs(g - g[idx, :]).max() / max(np.abs(g).max(), 1e-300))

    def cfl_dt(self, that, safety=0.5, dtmax=5e-3, dtmin=1e-10):
        psi, _ = self.stokes(that)
        ux, uz = self.velocity(psi)
        vmax = max(np.abs(ux).max(), np.abs(uz).max(), 1e-12)
        dzmin = float(np.diff(self.z).min())
        return float(np.clip(safety * min(self.Lx / self.Nx, dzmin) / vmax,
                             dtmin, dtmax))

    def run(self, that, tstop, dt=None, safety=0.5, dtmax=5e-3, cadence=200,
            verbose=True):
        t = 0.0
        hist = []
        while t < tstop:
            dtv = dt if dt is not None else self.cfl_dt(that, safety, dtmax)
            that = self.step(that, dtv)
            t += dtv
            if self.n_steps % cadence == 0:
                d = self.diagnostics(that)
                d['t'] = t
                d['dt'] = dtv
                hist.append(d)
                if verbose:
                    print('  t=%8.4f Nu=%9.5f/%9.5f urms=%8.4f umax=%8.3f '
                          'amp=%.3e dt=%.2e' % (t, d['Nu_top'], d['Nu_bot'],
                                                d['urms'], d['umax'], d['amp'],
                                                dtv), flush=True)
        return that, hist

    # ---------------------------------------------------------------- seeding
    def seed(self, amp=1e-3, m=1, n=1):
        """theta = amp cos(m pi x) sin(n pi z): mirror symmetric single roll."""
        X, Z = np.meshgrid(self.x, self.z, indexing='ij')
        return self.coeffs(amp * np.cos(m * np.pi * X) * np.sin(n * np.pi * Z))


def analytic_sigma(Ra, kx, kz):
    """Linear growth rate of the single mode cos(kx x) sin(kz z)."""
    g = kx * kx + kz * kz
    return -g + Ra * kx * kx / (g * g)


def analytic_stokes(Ra, kx, kz, Theta, x, z):
    """Exact solution for theta = Theta cos(kx x) sin(kz z)."""
    X, Z = np.meshgrid(x, z, indexing='ij')
    g = kx * kx + kz * kz
    th = Theta * np.cos(kx * X) * np.sin(kz * Z)
    psi = -Ra * kx / g ** 2 * np.sin(kx * X) * np.sin(kz * Z)
    uz = Ra * kx * kx / g ** 2 * th
    return th, psi, uz


# ------------------------------------------------------------------ self test
if __name__ == '__main__':
    import sys
    Ra, Nx, Nz = 1e4, 64, 64
    for zs in ('tau', 'galerkin'):
        m = RBC(Ra=Ra, Nx=Nx, Nz=Nz, zsolver=zs)
        kx = kz = np.pi
        th, psi_ex, uz_ex = analytic_stokes(Ra, kx, kz, 1.0, m.x, m.z)
        that = m.coeffs(th)
        psi, om = m.stokes(that)
        psi_n = m.grid(psi)
        ux, uz = m.velocity(psi)
        rel = lambda a, b: np.abs(a - b).max() / np.abs(b).max()
        print('[%s] single-mode Stokes  Ra=%g  kx=kz=pi' % (zs, Ra))
        print('    psi rel err = %.3e   uz rel err = %.3e' %
              (rel(psi_n, psi_ex), rel(uz, uz_ex)))
        print('    free-slip |psi(z=0,1)|      = %.3e' % max(np.abs(psi_n[:, 0]).max(),
                                                              np.abs(psi_n[:, -1]).max()))
        dpzz = m.grid((m.D @ (m.D @ psi.T)).T)
        print('    free-slip |psi_zz(z=0,1)|   = %.3e' % max(np.abs(dpzz[:, 0]).max(),
                                                              np.abs(dpzz[:, -1]).max()))
        print('    <uz,theta>/<theta,theta>   = %+.6f   (analytic %+.6f)' %
              ((uz * th).mean() / (th * th).mean(), Ra * kx ** 2 / (kx ** 2 + kz ** 2) ** 2))
        print('    mirror error               = %.3e' % m.mirror_error(that))
        # x = 0 and x = Lx are the same periodic node, so free-slip at the
        # side wall is a single condition on grid index 0.
        psixx = m.grid(-(m.kx[:, None] ** 2) * psi)
        print('    |psi_xx(x=0)|              = %.3e' % np.abs(psixx[0, :]).max())
        print('    sigma_analytic(kx=kz=pi)   = %+.6f' % analytic_sigma(Ra, kx, kz))
