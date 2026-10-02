"""Lagged viscosity + Picard defect correction for the coupled (all-wavenumber)
variable-viscosity Stokes solve.

Context
------------------------
The Schur-complement solver established that the coupled Stokes operator can be solved by
FGMRES + a block Schur-complement preconditioner built from eta_bar(z), but that
the whole matrix/preconditioner must be rebuilt whenever eta changes, i.e. on
every time step, because eta = eta(T(x,z,t)).  At 32x48 that costs

    assemble (Schur form) + dense LU + preconditioner  ~ 0.10 s
    each FGMRES iteration                              ~ 2-3 ms

whereas the *decoupled* depth-only Stokes solve is ~0.5 ms.  The dominant cost is
therefore the per-step rebuild, not the Krylov iteration.

Two ideas are implemented and separated here:

  (i)  LAGGED VISCOSITY ("naive lag").  Freeze eta at eta_lag and solve
       A(eta_lag) x = b(theta^n).  Only a back substitution per step, but the
       answer solves the wrong operator: it carries a splitting error
       proportional to the eta drift.  `solve_lag`.

  (ii) PICARD DEFECT CORRECTION.  Keep the frozen factorisation of (i) but
       iterate against the TRUE operator:

           x_0 = A_lag^{-1} b
           x_{k+1} = x_k + A_lag^{-1} (b - A(eta_new) x_k)          (*)

       This is stationary iterative refinement (a Richardson/Picard iteration
       preconditioned by the frozen operator).  It converges to the solution of
       A(eta_new) x = b, so the splitting error of (i) is REMOVED, not only
       reduced; the lag only costs iterations.  `solve_defect`.

The correction (*) needs A(eta_new) applied to a vector, and that is where the
`FreeSchurOperator` below comes in: it reproduces the action of the assembled
Schur matrix to machine precision in O(Nk * Nz^2) instead of O(Nk^2 * Nz^3),
i.e. ~50x cheaper than assembling.  Without it the defect correction would have
to assemble the true operator every step and nothing would be saved.

Exactness of the matrix-free operator
-------------------------------------
The assembled Schur matrix is (stokes_coupled.CoupledStokes.assemble_schur)

    A[i,j] = Lm_i M_{i-j} + [2 k_j^2 D2 M_{i-j} + 2 k_i^2 M_{i-j} D2
                            - 4 k_i k_j D M_{i-j} D] G_j

with M_d = multiplication by the d-th horizontal Fourier mode of eta, Lm_i =
D^2 - k_i^2, G_j = Dirichlet Green operator of Lm_j.  Summing over j and using
sum_j M_{i-j} f_j = (M f)_i for any mode-dependent vector f gives

    (A v)_i = Lm_i (M v)_i + D2 (M (2 k^2 G v))_i + 2 k_i^2 (M D2 (G v))_i
              - 4 k_i (D M (k G v)_z)_i

which is what `FreeSchurOperator.matvec` evaluates; `self_test` checks it
against the assembled matrix (expect ~1e-15).  In the last term the
factor k_j sits INSIDE the mode sum, so it must be applied to the field
(K G v)_j = k_j (G v)_j before multiplication by eta; only then can k_i be
factored out.  (Getting this wrong gives an O(1) error, caught by self_test.)

ADDENDUM: the conjugate half of the x-convolution
-----------------------------------------------------------
`sum_j M_{i-j} f_j` above is only the `n >= 0` half of the one-sided rfft
convolution.  The full contraction of the real, x-even eta with a real field f
also carries the negative-wavenumber half, which in the one-sided
representation is `eps_c * sum_{m>0} M_{i+m} conj(f_m)`: `eps_c = +1` for an
even mode-index factor (1, k^2) and `eps_c = -1` for an odd one (k), and
`conj(f_m) = partner_sign * f_m` with `partner_sign = -1` for the ODD-parity
Stokes field.  Of the four pieces of `matvec` only the cross term
`-4 k D M D Gv` carries an odd factor, so it alone uses `flip=True`.  Dropping
the half entirely, the previously shipped behaviour, left a ~12 %
operator error whenever eta had an x-spectrum; the internal cross-checks pin
the structure down against exact pointwise products.
"""
import numpy as np
import scipy.linalg as sla

from .stokes_coupled import (CoupledStokes, BlockSchurPrecond,
                             SchurOnlyPrecond, cheb_mult_complex)
from .vv_stokes import cheb_mult_matrix
from ..core.krylov_fgmres import fgmres

__all__ = ["FreeSchurOperator", "LaggedStokes", "self_test"]


def cheb_mult_matrix_fast(a, N):
    """Vectorised equivalent of `vv_stokes.cheb_mult_matrix` (exact same matrix).

    The reference implementation fills M[p, n] with a double python loop; here
    the three contributions

        a[p-n]  (p >= n)      a[n+p]  (n+p < N)      a[n-p]  (n >= p > 0)

    are written with boolean masks.  Identical output (checked in `self_test`),
    ~100x faster, which matters because the matrix-free coupled operator needs
    one such matrix per horizontal Fourier mode of eta.
    """
    a = np.asarray(a, dtype=float)
    idx = np.arange(N)
    P, Q = np.meshgrid(idx, idx, indexing='ij')
    M = np.zeros((N, N))
    m1 = P >= Q
    M[m1] += 0.5 * a[P[m1] - Q[m1]]
    m2 = (P + Q) < N
    M[m2] += 0.5 * a[P[m2] + Q[m2]]
    m3 = (P > 0) & (Q >= P)
    M[m3] += 0.5 * a[Q[m3] - P[m3]]
    return M


def cheb_mult_complex_fast(a, N):
    a = np.asarray(a)
    M = cheb_mult_matrix_fast(a.real, N).astype(complex)
    if np.abs(a.imag).max() > 0.0:
        M = M + 1j * cheb_mult_matrix_fast(a.imag, N)
    return M


class FreeSchurOperator:
    """Matrix-free application of the coupled Schur operator A(eta(x,z)).

    `partner_sign` is the sign of the negative-wavenumber ("conjugate") half of
    the one-sided rfft x-convolution; see the module docstring of
    `stokes_coupled.py`.  It MUST agree with `CoupledStokes.assemble_schur`, so
    it is inherited from `op_template` when one is given (default -1, the
    physical odd-parity Stokes class).
    """

    def __init__(self, model, eta_grid, op_template=None, partner_sign=None):
        self.m = model
        self.N = int(model.Nz)
        self.Nk = int(model.Nk)
        self.kx = np.asarray(model.kx, dtype=float)
        self.D = model.D
        self.D2 = self.D @ self.D
        self.V = model.V
        self.I = np.eye(self.N)
        if partner_sign is None:
            partner_sign = getattr(op_template, 'partner_sign', -1.0) \
                if op_template is not None else -1.0
        self.partner_sign = float(partner_sign)
        # eta in Chebyshev coefficients per x-mode: (Nk, N)
        self.eta_c = model.coeffs(eta_grid)
        self._Mc = {}
        if op_template is not None:
            self._G = [op_template.green(i) for i in range(self.Nk)]
        else:
            op = CoupledStokes(model, eta_grid)
            self._G = [op.green(i) for i in range(self.Nk)]
        self.n_mv = 0

    def _M(self, d):
        if abs(d) >= self.Nk:
            return None
        M = self._Mc.get(d)
        if M is None:
            a = self.eta_c[d] if d >= 0 else np.conj(self.eta_c[-d])
            M = cheb_mult_complex_fast(a, self.N)
            self._Mc[d] = M
        return M

    def mult_eta(self, f, flip=False):
        """(eta f) on the one-sided rfft array `f`, both halves included.

        The direct part is the `n >= 0` sum `sum_d M_d f_{i-d}` (exact Chebyshev
        product in z).  The negative-wavenumber half is `eps_c * sum_{m>0}
        M_{i+m} f_m`, where `eps_c = +1` for an even mode-index factor (1, k^2)
        and `-1` for an odd one (k); it is supplied through
        `flip`.  `self.partner_sign` is the parity of the field (`conj(f_m) =
        partner_sign * f_m`), default -1 (the odd Stokes class).

        See the module docstring of `stokes_coupled.py` for the derivation.
        Only the `-4 k_i k_j D M D` piece of the Schur operator carries an odd
        factor, which is why `matvec` calls this with `flip=True` exactly once.
        """
        Nk, N = self.Nk, self.N
        out = np.zeros((Nk, N), dtype=complex)
        for d in range(-(Nk - 1), Nk):
            M = self._M(d)
            if M is None:
                continue
            if d >= 0:
                out[d:] += f[:Nk - d] @ M.T
            else:
                out[:Nk + d] += f[-d:] @ M.T
        sgn = -self.partner_sign if flip else self.partner_sign
        if sgn:
            for d in range(1, Nk):
                M = self._M(d)
                if M is None:
                    continue
                out[:d] += sgn * (f[1:d + 1][::-1] @ M.T)
        return out

    def green(self, w):
        """psi = G w, one Dirichlet solve of Lm per wavenumber."""
        out = np.zeros_like(w)
        for i in range(self.Nk):
            out[i] = self._G[i] @ w[i]
        return out

    def matvec(self, v):
        v = np.asarray(v).reshape(self.Nk, self.N)
        w = self.green(v)                       # w = G v
        k2 = (self.kx ** 2)[:, None]
        t1 = self.mult_eta(v)
        t2 = self.mult_eta(2.0 * k2 * w)
        t3 = self.mult_eta((self.D2 @ w.T).T)
        # the cross piece carries an odd mode-index factor k_j, so its
        # conjugate half flips sign: `flip=True`
        t4 = self.mult_eta(self.kx[:, None] * (self.D @ w.T).T, flip=True)
        out = (self.D2 @ t1.T).T - k2 * t1                     # Lm (M v)
        out = out + (self.D2 @ t2.T).T                         # D2 M (2k^2 Gv)
        out = out + 2.0 * k2 * t3                              # 2k^2 M D2 Gv
        out = out - 4.0 * self.kx[:, None] * (self.D @ t4.T).T  # -4k D M D Gv
        # tau rows: v(z=0) = v(z=1) = 0
        out[:, -2] = v @ self.V[0]
        out[:, -1] = v @ self.V[-1]
        self.n_mv += 1
        return out.ravel()


class LaggedStokes:
    """Frozen-factorisation Stokes solver with optional Picard defect correction.

    `form='schur'` uses the Nk*Nz Schur-complement system.
    """

    def __init__(self, model, form='schur', rtol=1e-8, restart=80,
                 maxiter=400):
        self.m = model
        self.form = form
        self.rtol = float(rtol)
        self.restart = int(restart)
        self.maxiter = int(maxiter)
        # frozen state
        self.op_lag = None
        self.A_lag = None
        self.lu_lag = None
        self.P_lag = None
        self.eta_lag = None
        self.n_refresh = 0
        # timings (seconds, last call)
        self.t_assemble = 0.0
        self.t_lu = 0.0
        self.t_precond = 0.0

    # ------------------------------------------------------------- expensive
    def _build_precond(self):
        """The preconditioner of the FROZEN operator (cheap: Nk small LUs)."""
        if self.form == 'schur':
            if self._tala():
                from .stokes_tala import TalaSchurPrecond
                return TalaSchurPrecond(self.op_lag)
            return SchurOnlyPrecond(self.op_lag)
        if self._tala():
            raise NotImplementedError(
                "the TALA operator has no assembled 'mixed' form: use "
                "form='schur' (the mixed block preconditioner is a Boussinesq "
                "validation device)")
        return BlockSchurPrecond(self.op_lag, variant='full', schur='exact')

    def _tala(self):
        """True when this model runs the TALA operator."""
        return getattr(self.m, 'compressibility', 'boussinesq') == 'tala'

    def _op(self, eta_grid):
        """The Stokes operator class the model selects."""
        if self._tala():
            from .stokes_tala import TalaStokes
            return TalaStokes(self.m, eta_grid)
        return CoupledStokes(self.m, eta_grid)

    def refresh(self, eta_grid, keep_dense=False):
        """Assemble + factorise the FROZEN operator.

        `A_lag` (a dense `(Nk Nz)^2` complex matrix: 160 MB at 64x96) and
        `P_lag` are only used by the STALE-preconditioner FGMRES route
        (`solve_fgmres(stale=True)`), never by the defect correction, which
        needs only `lu_lag`.  Keeping them alive made every refresh allocate
        160 MB.  They are now built lazily
        (`_get_frozen_precond`) unless `keep_dense=True` is requested.
        """
        import time
        t0 = time.perf_counter()
        op = self._op(eta_grid)
        A = op.assemble_schur()
        self.t_assemble = time.perf_counter() - t0
        t0 = time.perf_counter()
        lu = sla.lu_factor(A)
        self.t_lu = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.op_lag = op
        self.lu_lag = lu
        self.A_lag = A if keep_dense else None
        self.P_lag = self._build_precond() if keep_dense else None
        self.t_precond = time.perf_counter() - t0
        self.eta_lag = np.asarray(eta_grid, dtype=float)
        self.n_refresh += 1
        return self

    def _get_frozen_precond(self):
        """Lazily build (and cache) the preconditioner of the frozen operator."""
        if self.P_lag is None:
            self.P_lag = self._build_precond()
        return self.P_lag

    def rhs(self, that):
        return self.op_lag.rhs_schur(that)

    def backsub(self, b, x0=None):
        """One application of A_lag^{-1} (the cheap per-step operation)."""
        if x0 is None:
            return sla.lu_solve(self.lu_lag, b)
        return x0 + sla.lu_solve(self.lu_lag, b)

    # ------------------------------------------------------ the three solvers
    def solve_lag(self, that):
        """(i) naive lag: solve with the FROZEN operator (has a split error)."""
        b = self.rhs(that)
        return sla.lu_solve(self.lu_lag, b), b

    def solve_defect(self, that, eta_new, K=2, x0=None):
        """(ii) Picard defect correction against the TRUE operator eta_new.

        Returns (x, b, res_hist) with res_hist = ||b - A_new x_k|| / ||b||
        BEFORE each correction (so res_hist[0] is the naive-lag residual).
        """
        b = self.rhs(that)
        if x0 is None:
            x = sla.lu_solve(self.lu_lag, b)
        else:
            x = np.array(x0, copy=True)
        Anew = FreeSchurOperator(self.m, eta_new, op_template=self.op_lag)
        bn = max(np.linalg.norm(b), 1e-300)
        hist = []
        for _ in range(K):
            r = b - Anew.matvec(x)
            hist.append(float(np.linalg.norm(r) / bn))
            x = x + sla.lu_solve(self.lu_lag, r)
        r = b - Anew.matvec(x)
        hist.append(float(np.linalg.norm(r) / bn))
        return x, b, hist

    def solve_fgmres(self, that, eta_new=None, stale=False, x0=None):
        """FGMRES on the TRUE operator with the frozen (or fresh) preconditioner."""
        op_new = self._op(eta_new)
        A = op_new.assemble_schur()
        b = op_new.rhs_schur(that)
        if stale:
            P = self._get_frozen_precond()
        elif self._tala():
            from .stokes_tala import TalaSchurPrecond
            P = TalaSchurPrecond(op_new)
        else:
            P = SchurOnlyPrecond(op_new)
        x, info, it, hist = fgmres(lambda v: A @ v, b, M=P.apply, x0=x0,
                                   rtol=self.rtol, restart=self.restart,
                                   maxiter=self.maxiter)
        return x, b, A, it, info

    def solve_direct(self, that, eta_new):
        op_new = self._op(eta_new)
        A = op_new.assemble_schur()
        b = op_new.rhs_schur(that)
        lu = sla.lu_factor(A)
        return sla.lu_solve(lu, b), b, A, lu


# ------------------------------------------------------------------ self test
def self_test(Nx=16, Nz=24, deta_T=100.0, verbose=True):
    from ..models.convection_modes import RBCVariableViscosity

    m = RBCVariableViscosity(Nx=Nx, Nz=Nz, deta_T=deta_T, symmetrize=True)
    rng0 = np.random.default_rng(11)
    af = rng0.normal(size=Nz)
    e0 = np.abs(cheb_mult_matrix_fast(af, Nz)
                - cheb_mult_matrix(af, Nz)).max()
    if verbose:
        print('  fast Chebyshev multiplication matrix vs vv_stokes: %.2e  %s'
              % (e0, 'OK' if e0 == 0.0 else 'FAIL'))
    X, Z = np.meshgrid(m.x, m.z, indexing='ij')
    rng = np.random.default_rng(4)
    theta = 0.3 * np.cos(np.pi * X) * np.sin(np.pi * Z) \
        + 0.1 * np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Z)
    eta = np.exp(-np.log(deta_T) * np.clip(1.0 - Z + theta, 0.0, 2.0))
    op = CoupledStokes(m, eta)
    A = op.assemble_schur()
    free = FreeSchurOperator(m, eta, op_template=op)
    v = rng.normal(size=m.Nk * m.Nz) + 1j * rng.normal(size=m.Nk * m.Nz)
    e = np.linalg.norm(A @ v - free.matvec(v)) / max(np.linalg.norm(A @ v), 1e-300)
    ok = (e < 1e-12) and (e0 == 0.0)
    if verbose:
        print('  matrix-free Schur matvec vs assembled A: rel err %.2e  %s'
              % (e, 'OK' if ok else 'FAIL'))
        # eta = eta_bar(z): the frozen operator must be block diagonal and the
        # defect correction must converge in ONE step (A_lag == A_new)
        eb = np.exp(-np.log(deta_T) * (1.0 - Z))
        opb = CoupledStokes(m, eb)
        ls = LaggedStokes(m).refresh(eb)
        that = m.seed(amp=1e-2)
        x0 = ls.solve_lag(that)[0]
        x1, b, hist = ls.solve_defect(that, eb, K=1, x0=x0)
        xd, _, _, lu = ls.solve_direct(that, eb)
        e2 = np.linalg.norm(x1 - xd) / max(np.linalg.norm(xd), 1e-300)
        ok &= e2 < 1e-10
        print('  defect correction with eta_lag == eta_new: rel err %.2e  %s'
              % (e2, 'OK' if ok else 'FAIL'))
        # 3. the stale-preconditioner FGMRES route must still work after the
        #    change that made `A_lag` / `P_lag` lazy
        ls2 = LaggedStokes(m).refresh(eta)
        assert ls2.A_lag is None and ls2.P_lag is None
        that2 = m.seed(amp=1e-2)
        xf, _, _, itf, infof = ls2.solve_fgmres(that2, eta, stale=True)
        xd2, _, _, _ = ls2.solve_direct(that2, eta)
        e3 = np.linalg.norm(xf - xd2) / max(np.linalg.norm(xd2), 1e-300)
        ok &= (e3 < 1e-6) and ls2.P_lag is not None
        print('  stale-precond FGMRES (lazy P_lag): rel err %.2e  iters %d '
              'info %d  %s' % (e3, itf, infof, 'OK' if e3 < 1e-6 else 'FAIL'))
        print('  lagged_stokes self_test %s' % ('PASSED' if ok else 'FAILED'))
    return bool(ok)


if __name__ == '__main__':
    print('lagged_stokes self test')
    self_test()
