"""Variable-viscosity elliptic Stokes solver (2D, infinite Prandtl, free slip).

Derivation of the operator (this is the only piece of physics in this file)
---------------------------------------------------------------------------
Streamfunction convention (identical to `rbc_galerkin.py`):

    u  = ( d psi/dz , -d psi/dx ),     omega = -lap(psi),   lap = d2/dx2 + d2/dz2

Momentum equation with a *variable* viscosity eta(x,z) (infinite Prandtl,
Boussinesq, unit reference viscosity, Ra explicit in the buoyancy):

    -d_j [ eta ( d_i u_j + d_j u_i ) ] + d_i p = Ra theta delta_{iz}        (M)

For constant eta, (M) reduces to -eta lap(u) + grad p = Ra theta e_z, whose
curl is  lap(omega) = -Ra d(theta)/dx, exactly the equation validated against
Blankenbach case 1a.

Taking the curl of (M) for general eta(z) and using the streamfunction
representation (tau_xz = eta (psi_zz - psi_xx), tau_xx - tau_zz = 4 eta psi_xz)
gives, per horizontal wavenumber k (d/dx -> i k, d/dz -> D):

    (D^2 + k^2)[ eta (D^2 + k^2) psi ] - 4 k^2 D[ eta D psi ] = i k Ra theta_hat
                                                                             (VV)

Two checks that fix every sign:
  * eta = 1:  (D^2+k^2)^2 - 4k^2 D^2 = (D^2 - k^2)^2, so (VV) becomes
    (D^2-k^2)^2 psi = i k Ra theta_hat, identical to the validated
    isoviscous operator (rbc_galerkin solves it as two 2nd-order tau steps).
  * eta = c (constant): (VV) is c (D^2-k^2)^2 psi = i k Ra theta_hat, so
    psi -> psi/c.  This is the exact viscosity-rescaling law and is used as an
    independent unit test of the operator.

Boundary conditions
-------------------
Free slip on z = 0, 1: impermeable  psi = 0, and zero tangential traction
    tau_xz = eta (psi_zz - psi_xx) = 0  ->  (D^2 + k^2) psi = 0.
With psi = 0 on the wall the second condition is psi_zz = 0, *independent of
eta*.  Both are imposed strongly by tau lines, so the four conditions are

    psi(0) = psi(1) = psi''(0) = psi''(1) = 0.

Side walls are free slip / insulating by mirror symmetry: the 1x1 box is
carried on a periodic grid of length Lx = 2 (see rbc_galerkin.py).

Discretisation
--------------
Chebyshev coefficients on [0,1] (the same basis as `cheb_D` in
rbc_galerkin.py).  D is strictly upper triangular, so (VV) is assembled
exactly in coefficient space:

    A(k) = P0 + k^2 P1 + k^4 M
    P0   = D2 M D2
    P1   = D2 M + M D2 - 4 D M D
    M    = multiplication-by-eta matrix          (exact Galerkin product)

Rows 0..N-5 carry the differential equation, rows N-4..N-1 are replaced by the
four free-slip boundary rows (the classical Chebyshev tau layout).  The
operator does not depend on time, so each wavenumber is LU-factorised ONCE and
every later Stokes solve is a single back substitution (`_CachedSolveMixin`
from rbc_galerkin.py, the same cache architecture as the isoviscous solver).

This is a DIRECT solve.  There is no initial-guess input, hence no warm start
and no headroom, exactly as for the isoviscous operator.

Conditioning and the reduced form
---------------------------------
The 4th-order tau layout above replaces FOUR equation rows by boundary rows and
its condition number grows like ~N^8 (Nz = 96 -> ~2.4e14, within a factor ~100
of the double-precision wall).  `VVStokesReduced` below therefore assembles the
*block-eliminated* 2nd-order form (u = (D^2+k^2) psi, two boundary rows), whose
conditioning falls back to ~N^4.  Both forms solve the same continuum problem;
the reduced one is the default in `convection_modes.py` and the 4th-order one
is kept as a validated reference.
"""
import numpy as np
import scipy.linalg as sla

from ..core.rbc_galerkin import _CachedSolveMixin


def cheb_mult_matrix(a, N):
    """Exact Chebyshev-coefficient matrix of multiplication by

        eta(z) = sum_j a_j T_j(2z-1).

    Uses T_m T_n = (T_{m+n} + T_{|m-n|})/2.  The coefficient of T_p in
    eta * psi is

        (1/2) sum_{j,n} a_j c_n [ delta_{j+n,p} + delta_{|j-n|,p} ],

    i.e. every (j, n) pair contributes 1/2 twice when both deltas fire.  The
    only degenerate case is j+n = |j-n| = p, which happens for n = 0, j = 0 or
    p = 0; the code below reproduces the double counting correctly (the result
    is the identity for a = e_0, verified in `self_test`).
    """
    a = np.asarray(a, dtype=float)
    assert a.shape == (N,), "eta coefficients must have length N"
    M = np.zeros((N, N))
    for p in range(N):
        for n in range(N):
            s = 0.0
            if p >= n:
                s += a[p - n]
            if n + p < N:
                s += a[n + p]
            if p > 0 and n >= p:
                s += a[n - p]
            M[p, n] = 0.5 * s
    return M


class VVStokes(_CachedSolveMixin):
    """Per-wavenumber direct solver of (VV) for a frozen eta(z)."""

    def __init__(self, Z, kx, Ra, eta_c, use_lu=True):
        self.Z = Z
        self.N = int(Z.N)
        self.z = Z.z
        self.V = Z.V
        self.D = Z.D
        self.D2 = Z.D @ Z.D
        self.kx = np.asarray(kx, dtype=float)
        self.Ra = float(Ra)
        self.eta_c = np.asarray(eta_c, dtype=float)
        self.M = cheb_mult_matrix(self.eta_c, self.N)
        D, D2, M = self.D, self.D2, self.M
        self.P0 = D2 @ M @ D2
        self.P1 = D2 @ M + M @ D2 - 4.0 * (D @ M @ D)
        # boundary rows: psi(0), psi(1), psi''(0), psi''(1)
        self.bc = np.vstack([self.V[0, :], self.V[-1, :],
                             (self.V @ D2)[0, :], (self.V @ D2)[-1, :]])
        self._init_cache(use_lu)
        for k in self.kx:
            if k != 0.0:
                self._factors_for(k, 0.0)

    # ------------------------------------------------------------- assembly
    def _assemble(self, k, lam=0.0):
        k2 = k * k
        A = self.P0 + k2 * self.P1 + (k2 * k2) * self.M
        A[-4:, :] = self.bc
        return A

    def solve(self, k, that):
        """theta_hat (Chebyshev coefficients, one wavenumber) -> (psi, omega).

        Returns (psi_hat, omega_hat) with omega = -lap(psi).
        """
        b = np.array(1j * float(k) * self.Ra * np.asarray(that, dtype=complex))
        b[-4:] = 0.0
        lu = self._factors_for(k, 0.0)
        if lu is None:
            A = self._assemble(k, 0.0)
            psi = (sla.solve(A, b.real) + 1j * sla.solve(A, b.imag))
        else:
            psi = sla.lu_solve(lu, b.real) + 1j * sla.lu_solve(lu, b.imag)
        om = -(self.D2 - (float(k) ** 2) * np.eye(self.N)) @ psi
        return psi, om

    # ---------------------------------------------------------- diagnostics
    def residual(self, k, that, psi):
        """Relative residual of the ODE rows and of the four BC rows.

        Returns (rel_ode, max_abs_bc).  `rel_ode` is measured only on rows
        0..N-5 (the rows that carry the equation); the BC rows are
        checked separately against psi(0), psi(1), psi''(0), psi''(1).
        """
        k2 = float(k) ** 2
        A = self.P0 + k2 * self.P1 + (k2 * k2) * self.M
        b = np.array(1j * float(k) * self.Ra * np.asarray(that, dtype=complex))
        r = A[:self.N - 4] @ psi - b[:self.N - 4]
        scale = max(np.abs(b).max(), 1e-300)
        rel = float(np.abs(r).max() / scale)
        vals = np.array([(self.V[0] @ psi).real, (self.V[-1] @ psi).real,
                         ((self.V @ self.D2)[0] @ psi).real,
                         ((self.V @ self.D2)[-1] @ psi).real])
        return rel, float(np.abs(vals).max())

    def cond(self, k):
        return float(np.linalg.cond(self._assemble(k, 0.0)))


class VVStokesReduced(_CachedSolveMixin):
    """Reduced (2nd-order) variable-viscosity Stokes solver via block elimination.

    Motivation
    ----------
    The 4th-order tau operator assembled by `VVStokes` replaces FOUR equation
    rows by boundary rows.  In coefficient space its condition number grows
    like ~N^8 (measured on the frozen production state), which at Nz = 96 is
    ~2.4e14, within a factor ~100 of the double-precision wall.  This is a
    property of the 4th-order Chebyshev-tau discretisation, not of the physics,
    and it caps Nz independently of deta_T.

    WHICH eliminant works
    ---------------------------------------------
    The obvious choice  u := (D^2 + k^2) psi  DOES NOT WORK: on the Fourier
    wavenumbers of this solver kx = m*pi, the operator (D^2 + k^2) with
    Dirichlet data has the exact null vector sin(k z) (sin(m pi z) = 0 at both
    walls).  Its tau matrix is therefore singular, measured cond ~ 1e18 at
    Nz = 24, and `sla.solve` reports rcond ~ 1e-24.  Any "N^4 conditioning"
    claim built on that form is an artefact; the operator does not exist.

    The correct eliminant is the vorticity itself.  Writing

        L- := D^2 - k^2,      u := L- psi  =  -omega  =  -(D^2 - k^2) psi

    (so u = -omega is exactly the field the isoviscous solver computes first),
    the operator identity

        (D^2+k^2) M (D^2+k^2) - 4 k^2 D M D  =  L- M L- + 2 k^2 [D,[D,M]]

    (M = multiplication by eta; [D,[D,M]] = D^2 M - 2 D M D + M D^2 is the
    commutator, which in the continuum is multiplication by eta''(z)) turns
    (VV) into the 2nd-order problem

        A u := L- [ eta u ] + 2 k^2 [D,[D,M]] G u = i k Ra theta ,
        u(0) = u(1) = 0 ,        psi = G u ,

    where G is the Dirichlet Green operator of L-.  Because psi = 0 on the
    walls, u(0) = L- psi = psi''(0), so u(0)=u(1)=0 is exactly the free-slip
    condition psi''(0)=psi''(1)=0; G supplies psi(0)=psi(1)=0.  All four
    free-slip conditions are thus imposed.

    `L-` with Dirichlet data has NO null space (its homogeneous solutions are
    e^{+-k z}, which cannot vanish at both ends), so G is well conditioned,
    it is the same operator `rbc_galerkin.ChebTau` already uses for
    the validated isoviscous solver.  The assembled A is a 2nd-order tau
    operator (two boundary rows), so its conditioning is expected to be ~N^4.

    eta = 1 (M = I, M'' = 0) reduces A to L-, and psi = G u = G^2 (i k Ra theta),
    i.e. exactly the validated two-step isoviscous solve.

    Discretisation
    --------------
    G is materialised once per wavenumber (`_green`) from the *same* tau layout
    as `ChebTau` (rows 0..N-3 carry L-, rows N-2,N-1 are psi(0)=psi(1)=0, last
    two right-hand sides zero).  A(k) is therefore still per-wavenumber and
    still cached exactly like the 4th-order form.

    This is a DIRECT solve: LU once per wavenumber, back substitution per step.
    No initial guess -> no warm start, no headroom (same as `VVStokes`).
    """

    def __init__(self, Z, kx, Ra, eta_c, use_lu=True):
        self.Z = Z
        self.N = int(Z.N)
        self.z = Z.z
        self.V = Z.V
        self.D = Z.D
        self.D2 = Z.D @ Z.D
        self.kx = np.asarray(kx, dtype=float)
        self.Ra = float(Ra)
        self.eta_c = np.asarray(eta_c, dtype=float)
        self.M = cheb_mult_matrix(self.eta_c, self.N)
        # Exact DISCRETE commutator  Comm = [D,[D,M]] = D2 M - 2 D M D + M D2.
        # In the continuum this is multiplication by eta''(z).  It is NOT
        # identical to cheb_mult_matrix(D2 @ eta_c) on the truncated coefficient
        # space: the truncated multiplication drops Chebyshev modes >= N, and
        # D2 of a degree-N mode feeds back into mode N-2, so the two differ by
        # a spectrally small truncation term (~1e-5 relative on arbitrary
        # coefficient vectors, ~1e-14 on resolved smooth fields).  Using the
        # matrix commutator below makes the block elimination an EXACT
        # identity of the discrete 4th-order operator instead of an
        # approximation to it.
        self.Comm = (self.D2 @ self.M - 2.0 * (self.D @ self.M @ self.D)
                     + self.M @ self.D2)
        # operator that zeroes the two tau rows of the right-hand side
        self._P = np.eye(self.N)
        self._P[-2:, :] = 0.0
        self._J = np.eye(self.N)
        self._H = {}
        # u(0) = u(1) = 0  (== free-slip psi'' = 0 because psi = 0 there)
        self.bc = np.vstack([self.V[0, :], self.V[-1, :]])
        self._init_cache(use_lu)
        for k in self.kx:
            if k != 0.0:
                self._factors_for(k, 0.0)

    # ------------------------------------------------------- Green operator
    def _green(self, k):
        """Matrix G(k) with psi = G u solving (D^2-k^2) psi = u, psi(0)=psi(1)=0.

        This is `ChebTau` at lam = 0 (the isoviscous solver's omega/psi
        operator).  `ChebTau.solve` zeroes rhs[-2:], so the linear map is
        u -> A^{-1} P u.
        """
        key = round(float(k), 10)
        G = self._H.get(key)
        if G is None:
            A = self.D2 - (float(k) ** 2) * self._J
            A[-2, :] = self.V[0, :]
            A[-1, :] = self.V[-1, :]
            G = sla.solve(A, self._P)
            self._H[key] = G
        return G

    def Lminus(self, k):
        return self.D2 - (float(k) ** 2) * self._J

    # ------------------------------------------------------------- assembly
    def _assemble(self, k, lam=0.0):
        k2 = float(k) ** 2
        G = self._green(k)
        A = self.Lminus(k) @ self.M + 2.0 * k2 * (self.Comm @ G)
        A[-2:, :] = self.bc
        return A

    def solve(self, k, that):
        """theta_hat (Chebyshev coefficients, one wavenumber) -> (psi, omega)."""
        b = np.array(1j * float(k) * self.Ra * np.asarray(that, dtype=complex))
        b[-2:] = 0.0
        lu = self._factors_for(k, 0.0)
        if lu is None:
            A = self._assemble(k, 0.0)
            u = sla.solve(A, b.real) + 1j * sla.solve(A, b.imag)
        else:
            u = sla.lu_solve(lu, b.real) + 1j * sla.lu_solve(lu, b.imag)
        psi = self._green(k) @ u
        om = -u                                  # omega = -L- psi = -u
        return psi, om

    # ---------------------------------------------------------- diagnostics
    def residual(self, k, that, psi):
        """Relative residual of the equation rows and the two u-BC rows."""
        k2 = float(k) ** 2
        G = self._green(k)
        A = self.Lminus(k) @ self.M + 2.0 * k2 * (self.Comm @ G)
        b = np.array(1j * float(k) * self.Ra * np.asarray(that, dtype=complex))
        u = self.Lminus(k) @ psi
        r = A[:self.N - 2] @ u - b[:self.N - 2]
        scale = max(np.abs(b).max(), 1e-300)
        rel = float(np.abs(r).max() / scale)
        vals = np.array([(self.V[0] @ u).real, (self.V[-1] @ u).real])
        return rel, float(np.abs(vals).max())

    def cond(self, k):
        return float(np.linalg.cond(self._assemble(k, 0.0)))

    @property
    def order(self):
        return 'reduced-2nd'


# ---------------------------------------------------------------- self test
def self_test(N=48, verbose=True):
    """Cheap structural tests of the operator, independent of any flow run."""
    from ..core.rbc_galerkin import cheb_D, cgl, cheb_V
    z = cgl(N)
    V = cheb_V(z, N)
    Vi = np.linalg.inv(V)
    D = cheb_D(N)
    ok = True

    # 1. multiplication matrix is the identity for eta = 1
    M1 = cheb_mult_matrix(np.eye(N)[0], N)
    e = np.abs(M1 - np.eye(N)).max()
    ok &= e < 1e-14
    if verbose:
        print('  [1] ||M(eta=1) - I||_max           = %.3e' % e)

    # 2. exact product on a random smooth pair, checked via grid quadrature:
    #    eta = T_2, psi = T_3  ->  eta*psi = (T_5 + T_1)/2
    a = np.zeros(N); a[2] = 1.0
    c = np.zeros(N); c[3] = 1.0
    p = cheb_mult_matrix(a, N) @ c
    want = np.zeros(N); want[1] = 0.5; want[5] = 0.5
    e2 = np.abs(p - want).max()
    ok &= e2 < 1e-14
    if verbose:
        print('  [2] T_2 * T_3  -> (T_5+T_1)/2      err = %.3e' % e2)

    # 3. random eta: grid product vs coefficient product, spectrally resolved
    rng = np.random.default_rng(0)
    ac = np.zeros(N)
    ac[:8] = rng.normal(size=8) * 0.1
    cc = np.zeros(N)
    cc[:8] = rng.normal(size=8) * 0.1
    eg = V @ ac
    pg = V @ cc
    coef_direct = Vi @ (eg * pg)
    coef_mat = cheb_mult_matrix(ac, N) @ cc
    e3 = np.abs(coef_direct - coef_mat).max()
    ok &= e3 < 1e-13
    if verbose:
        print('  [3] grid product vs M(eta) product  err = %.3e' % e3)

    # 4. eta = 1: the assembled operator equals (D^2-k^2)^2 in the eq. rows
    k = np.pi
    A = (D @ D - k * k * np.eye(N))
    A = A @ A
    kx = np.array([k])
    S = VVStokes.__new__(VVStokes)
    S.N = N; S.V = V; S.D = D; S.D2 = D @ D; S.kx = kx
    S.P0 = S.D2 @ M1 @ S.D2
    S.P1 = S.D2 @ M1 + M1 @ S.D2 - 4.0 * (D @ M1 @ D)
    S.M = M1
    S.bc = np.vstack([V[0], V[-1], (V @ S.D2)[0], (V @ S.D2)[-1]])
    B = S._assemble(k)
    e4 = (np.abs(B[:N - 4] - A[:N - 4]).max()
          / max(np.abs(A[:N - 4]).max(), 1e-300))
    ok &= e4 < 1e-12
    if verbose:
        print('  [4] ||A(eta=1) - (D^2-k^2)^2||_rel = %.3e  '
              '(eq. rows, ||.||_max=%.2e)' % (e4, np.abs(A[:N - 4]).max()))

    # 5. eta = 1: the reduced solve must reproduce the VALIDATED two-step
    #    isoviscous solve (rbc_galerkin.ChebTau: (D^2-k^2)om = f, then
    #    (D^2-k^2)psi = -om) to machine precision.  This is the meaningful
    #    equivalence test; a raw matrix identity A4@G == A_red cannot be
    #    checked in double precision because assembling A4 requires the very
    #    cancellation that makes the 4th-order operator ill-conditioned.
    from ..core.rbc_galerkin import ChebTau
    Zt = ChebTau(N)
    RAt = 1e4
    that = np.zeros(N, dtype=complex)
    that[0] = 1.0
    f = -1j * k * RAt * that
    om_ref = Zt.solve(k, f, 0.0)
    psi_ref = Zt.solve(k, -om_ref, 0.0)
    S2 = VVStokesReduced.__new__(VVStokesReduced)
    S2.N = N; S2.V = V; S2.D = D; S2.D2 = D @ D; S2.kx = kx
    S2.M = M1; S2.Comm = np.zeros((N, N)); S2.Ra = RAt
    S2._P = np.eye(N); S2._P[-2:, :] = 0.0
    S2._J = np.eye(N); S2._H = {}
    S2.bc = np.vstack([V[0], V[-1]])
    S2._init_cache(True)
    psi2, om2 = S2.solve(k, that)
    e5 = float(np.abs(psi2 - psi_ref).max() / max(np.abs(psi_ref).max(), 1e-300))
    e5o = float(np.abs(om2 - om_ref).max() / max(np.abs(om_ref).max(), 1e-300))
    ok &= e5 < 1e-12 and e5o < 1e-12
    if verbose:
        print('  [5] eta=1: reduced vs ChebTau 2-step  psi %.3e  omega %.3e'
              % (e5, e5o))

    # 6. the discrete commutator makes the elimination an exact identity of the
    #    discrete 4th-order operator: A4 = L- M L- + 2k^2 Comm for a
    #    NON-constant eta.  Also report how far Comm is from the (truncated)
    #    multiplication by eta'' on resolved smooth vectors.
    rng2 = np.random.default_rng(1)
    ec = np.zeros(N)
    ec[:6] = rng2.normal(size=6) * 0.1
    ec[0] += 1.0
    Me = cheb_mult_matrix(ec, N)
    Comm = D @ D @ Me - 2.0 * D @ Me @ D + Me @ D @ D
    Lp6 = D @ D + k * k * np.eye(N)
    Lm6 = D @ D - k * k * np.eye(N)
    e6 = float(np.abs(Lp6 @ Me @ Lp6 - 4.0 * k * k * (D @ Me @ D)
                      - (Lm6 @ Me @ Lm6 + 2.0 * k * k * Comm)).max()
               / max(np.abs(Lp6 @ Me @ Lp6).max(), 1e-300))
    Mpp = cheb_mult_matrix(D @ D @ ec, N)
    cs = np.zeros(N); cs[:5] = rng2.normal(size=5)     # smooth test vector
    e6b = float(np.abs((Comm - Mpp) @ cs).max() / max(np.abs(Comm @ cs).max(), 1e-300))
    ok &= e6 < 1e-12
    if verbose:
        print('  [6] ||A4 - (L- M L- + 2k^2 Comm)||_rel = %.3e   '
              '||(Comm-Mpp)c_smooth||_rel = %.1e' % (e6, e6b))

    if verbose:
        print('  self_test %s' % ('PASSED' if ok else 'FAILED'))
    return bool(ok)


if __name__ == '__main__':
    print('VVStokes operator self test (N=48)')
    self_test()
