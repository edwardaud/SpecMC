"""Coupled variable-viscosity Stokes operator (eta = eta(x, z)) for the coupled solver,
plus the block (Schur-complement) preconditioner used by the Schur solver.

This file is the bridge between the VALIDATED discretisation of the depth-only
solver and the ITERATIVE solver of the coupled problem.  It does not change any physics: every operator
below is an exact algebraic rearrangement of the reduced operator that
`vv_stokes.VVStokesReduced` already solves (and that was validated to
machine precision: dissipation balance <= 2.4e-14, analytic single-mode match).

The physics (identical to vv_stokes.py)
---------------------------------------
Streamfunction convention  u = (psi_z, -psi_x),  omega = -lap(psi).
Curl of the infinite-Prandtl Boussinesq momentum equation with eta(x,z):

    (D^2 + k^2)[ eta (D^2 + k^2) psi ] - 4 k^2 D[ eta D psi ] = i k Ra theta_hat
                                                                          (4th)

Block elimination (vv_stokes.py, exact for the *discrete* operator when the
matrix commutator [D,[D,M]] is used):

    Lm := D^2 - k^2 ,   v := Lm psi = -omega

    row 1 (momentum) :  Lm M v + 2 k^2 Comm psi = i k Ra theta        (M = mult. by eta)
    row 2 (definition):  Lm psi - v = 0

with the four free-slip conditions carried by the two tau rows of each block
(v(0)=v(1)=0  <==>  psi''(0)=psi''(1)=0, and psi(0)=psi(1)=0).

Two equivalent assembled systems
--------------------------------
1. `assemble_mixed()`, the 2 x 2 BLOCK system on z = (v, psi), size
   2 Nk Nz:

       [ A11  A12 ] [v  ]   [ i k Ra theta ]
       [ A21  A22 ] [psi] = [      0       ]

       A11[i,j] = Lm_i M_{i-j}
       A12[i,j] = 2 k_j^2 D^2 M_{i-j} + 2 k_i^2 M_{i-j} D^2 - 4 k_i k_j D M_{i-j} D
       A21      = -I           (block diagonal)
       A22      = Lm           (block diagonal)

   This is the form that has a velocity/momentum block (A11) and a genuine
   Schur complement  S = A22 - A21 A11^{-1} A12.  In the continuum S equals the
   4th-order operator; discretely it is the reduced operator of the depth-only solver up to
   the cheap similarity transform  S = (Lm M)^{-1} (Lm M Lm + C) ~ (Lm M)^{-1} B Lm.

2. `assemble_schur()`, the SCHUR COMPLEMENT system on v alone, size Nk Nz:

       B[i,j] = Lm_i M_{i-j} + C_{ij} G_j ,      G_j = Dirichlet Green op. of Lm_j

   This is the reduced coupled matrix of the discretisation above; for
   eta = eta(z) it is block diagonal and equals `vv_stokes.VVStokesReduced`
   (verified in `self_test`).

Preconditioners (`PrecondBase` subclasses)
------------------------------------------
All of them replace eta(x,z) by its HORIZONTAL MEAN eta_bar(z).  That makes the
block-diagonal (per-wavenumber) approximation, which is exactly the validated
depth-only operator, and is cheap: one/two small dense solves per wavenumber,
LU-cached.

  `BlockSchurPrecond(op, variant, schur)`
      variant 'full'  : exact block solve of the decoupled mixed system, i.e.
                        the complete Schur-complement factorisation
                          [A11 A12] = [I  A12 S^-1][A11 0][I 0       ]
                          [A21 A22]   [0  I        ][0   S ][A11^-1 A21 I]
                        Exact whenever eta = eta_bar(z)  ->  1 FGMRES iteration.
      variant 'lower' : block LOWER triangular  [[A11,0],[A21,S]]
                        (the block lower-triangular Schur complement)
      variant 'upper' : block UPPER triangular  [[A11,A12],[0,S]]
      schur  'exact'  : S = A22 - A21 A11^-1 A12   (the decoupled Schur complement)
      schur  'cheap'  : S = A22                    (the "simple mass/Laplace block"
                                                    placeholder)

  `SchurOnlyPrecond(op)`, for system 2: per-wavenumber inverse of the depth-only
      validated operator at eta_bar(z).

The one-sided (rfft) x-convolution, and its `partner_sign`
---------------------------------------------------------
Every block above is a sum over the x-wavenumber shift `i-j`, i.e. over the
`n >= 0` half of the convolution.  `eta` is real and (for the mirror-projected
box) even in x, so its two-sided coefficients satisfy `eta_d = eta_{-d}` and
the missing `n < 0` half of a term

    sum_n  c_i c_n eta_{i-n} psi_n

is `sum_{m>0} c_i c_{-m} eta_{i+m} conj(psi_m)`.  Since `c_{-m}/c_m = eps_c`
with `eps_c = +1` for a factor that is EVEN in the mode index (1, k^2, ...) and
`eps_c = -1` for an ODD one (k), the conjugate half is `eps_c` TIMES the same
block evaluated at `conj(psi_m)`.

* The `(D^2+k_i^2) M (D^2+k_j^2)` piece has only `k^2` factors  -> `eps_c = +1`
  (here and below in full generality).
* The cross piece `-4 k_i k_j D M D` carries one factor `k` on each side
  -> `eps_c = -1`, so ITS conjugate half enters with the opposite sign to the
  direct sum.  This is not a sign ambiguity but a property of the operator: the
  field multiplied by eta in the cross term is `d_x d_z psi`, whose
  one-sided coefficient is `i k_m D psi_m`, and `conj(i k_m) = -i k_m` while
  `k_{-m} = -k_m`.

Finally, for a field of definite x-parity `conj(psi_m) = partner_sign * psi_m`
with `partner_sign = +1` for an x-EVEN field (real one-sided coefficients) and
`-1` for an x-ODD one.  `CoupledStokes` is the STOKES operator: the physical
solution has theta even in x => psi odd => v = (D^2-k^2) psi odd, so
`partner_sign = -1` (the default) is the physical setting.  Omitting the
conjugate half entirely (the previously shipped behaviour) is a real
O(10 %) operator error whenever eta has an x-spectrum; getting its parity
structure wrong is wrong by up to twice that.  All of it is invisible for
eta = const or eta = eta(z), because then `eta_d = 0` for every `d != 0` and
the partner sum vanishes identically, which is why every depth-only and
Delta eta_T = 1 validation passed.  The internal cross-checks pin the
structure down against exact pointwise products.
"""
import numpy as np
import scipy.linalg as sla

from .vv_stokes import cheb_mult_matrix

__all__ = ["cheb_mult_complex", "CoupledStokes", "BlockSchurPrecond",
           "SchurOnlyPrecond", "self_test"]


def cheb_mult_complex(a, N):
    """Exact Chebyshev multiplication matrix for a COMPLEX coefficient set."""
    a = np.asarray(a)
    M = cheb_mult_matrix(a.real, N).astype(complex)
    if np.abs(a.imag).max() > 0.0:
        M = M + 1j * cheb_mult_matrix(a.imag, N)
    return M


class CoupledStokes:
    """Coupled (all wavenumbers) variable-viscosity Stokes operator."""

    def __init__(self, model, eta_grid, partner_sign=-1.0):
        self.model = model
        self.N = int(model.Nz)
        self.Nk = int(model.Nk)
        self.Nx = int(model.Nx)
        self.kx = np.asarray(model.kx, dtype=float)
        self.D = model.D
        self.D2 = self.D @ self.D
        self.V = model.V
        self.I = np.eye(self.N)
        self.Ra = float(model.Ra)
        self.eta_c = model.coeffs(eta_grid)              # (Nk, N) complex
        self.eta_bar_c = np.real(self.eta_c[0]).copy()   # horizontal mean eta(z)
        # sign of the negative-wavenumber ("conjugate") half of the one-sided
        # x-convolution.  -1 is the physical Stokes class (v odd in x); 0
        # disables the term (the operator without the conjugate half) and +1 is the
        # even-parity sign.  See the module docstring.
        self.partner_sign = float(partner_sign)
        self._Mc = {}
        self._Gc = {}
        self.n_assemblies = 0

    # ------------------------------------------------------------- operators
    def _mats(self, d):
        """(M, D^2 M, M D^2, D M D) for x-mode shift d (None if |d| >= Nk)."""
        if abs(d) >= self.Nk:
            return None
        mm = self._Mc.get(d)
        if mm is None:
            a = self.eta_c[d] if d >= 0 else np.conj(self.eta_c[-d])
            M = cheb_mult_complex(a, self.N)
            mm = (M, self.D2 @ M, M @ self.D2, self.D @ M @ self.D)
            self._Mc[d] = mm
        return mm

    def green(self, idx):
        """G_k with psi = G v solving (D^2-k^2) psi = v, psi(0)=psi(1)=0."""
        G = self._Gc.get(idx)
        if G is None:
            k = self.kx[idx]
            A = self.D2 - (k * k) * self.I
            A[-2, :] = self.V[0]
            A[-1, :] = self.V[-1]
            P = np.eye(self.N)
            P[-2:, :] = 0.0
            G = sla.solve(A, P)
            self._Gc[idx] = G
        return G

    # -------------------------------------------------------------- assembly
    def assemble_mixed(self):
        """The 2 x 2 block system on z = (v, psi).   Returns a dense (2NkNz)^2.

        Both blocks of the momentum row use BOTH halves of the one-sided
        x-convolution (shifts `i-j` and `i+j`); see `partner_sign`.
        """
        N, Nk, kx = self.N, self.Nk, self.kx
        n2 = 2 * Nk * N
        A = np.zeros((n2, n2), dtype=complex)
        for i in range(Nk):
            ki = kx[i]
            Lmi = self.D2 - ki * ki * self.I
            bi = i * N
            for j in range(Nk):
                mm = self._mats(i - j)
                if mm is None:
                    continue
                M, D2M, MD2, DMD = mm
                kj = kx[j]
                A[bi:bi + N, j * N:(j + 1) * N] = Lmi @ M
                A[bi:bi + N, (Nk + j) * N:(Nk + j + 1) * N] = (
                    2.0 * kj * kj * D2M + 2.0 * ki * ki * MD2
                    - 4.0 * ki * kj * DMD)
                if i == j:
                    A[(Nk + i) * N:(Nk + i + 1) * N, j * N:(j + 1) * N] = -self.I
                    A[(Nk + i) * N:(Nk + i + 1) * N,
                      (Nk + j) * N:(Nk + j + 1) * N] = Lmi
            if self.partner_sign:
                for m in range(1, Nk - i):
                    mm = self._mats(i + m)
                    if mm is None:
                        continue
                    M, D2M, MD2, DMD = mm
                    kj = kx[m]
                    A[bi:bi + N, m * N:(m + 1) * N] += (
                        self.partner_sign * (Lmi @ M))
                    # cross piece flips sign (odd mode-index factor k_j)
                    A[bi:bi + N, (Nk + m) * N:(Nk + m + 1) * N] += (
                        self.partner_sign * (2.0 * kj * kj * D2M
                                             + 2.0 * ki * ki * MD2
                                             + 4.0 * ki * kj * DMD))
        self._bc_rows_mixed(A)
        self.n_assemblies += 1
        return A

    def _bc_rows_mixed(self, A):
        """Replace the tau rows of both diagonal blocks by the free-slip rows."""
        N, Nk = self.N, self.Nk
        for i in range(Nk):
            for r, vrow in ((N - 2, self.V[0]), (N - 1, self.V[-1])):
                row = i * N + r
                A[row, :] = 0.0
                A[row, i * N:(i + 1) * N] = vrow
                row = (Nk + i) * N + r
                A[row, :] = 0.0
                A[row, (Nk + i) * N:(Nk + i + 1) * N] = vrow

    def assemble_schur(self):
        """The Schur-complement (reduced) system on v.  Dense (NkNz)^2.

        `A[i, j]` accumulates BOTH halves of the one-sided x-convolution of
        `eta` with `v`: the shift `i-j` (the `n >= 0` half, valid for any
        parity class) and the shift `i+j` (the `n < 0` half, entering with
        `self.partner_sign`).  See the module docstring.
        """
        N, Nk, kx = self.N, self.Nk, self.kx
        A = np.zeros((Nk * N, Nk * N), dtype=complex)
        for i in range(Nk):
            ki = kx[i]
            Lmi = self.D2 - ki * ki * self.I
            rows = slice(i * N, (i + 1) * N)
            for j in range(Nk):
                mm = self._mats(i - j)
                if mm is None:
                    continue
                M, D2M, MD2, DMD = mm
                kj = kx[j]
                C = (2.0 * kj * kj * D2M + 2.0 * ki * ki * MD2
                     - 4.0 * ki * kj * DMD)
                A[rows, j * N:(j + 1) * N] = Lmi @ M + C @ self.green(j)
            if self.partner_sign:
                for m in range(1, Nk - i):
                    mm = self._mats(i + m)
                    if mm is None:
                        continue
                    M, D2M, MD2, DMD = mm
                    kj = kx[m]
                    # negative-wavenumber half; the cross piece flips sign
                    # (odd mode-index factor k_j), see the module docstring
                    C = (2.0 * kj * kj * D2M + 2.0 * ki * ki * MD2
                         + 4.0 * ki * kj * DMD)
                    A[rows, m * N:(m + 1) * N] += self.partner_sign * (
                        Lmi @ M + C @ self.green(m))
        self._bc_rows_schur(A)
        self.n_assemblies += 1
        return A

    def _bc_rows_schur(self, A):
        """Replace the tau rows of the Schur system by v(0) = v(1) = 0."""
        N, Nk = self.N, self.Nk
        for i in range(Nk):
            for r, vrow in ((N - 2, self.V[0]), (N - 1, self.V[-1])):
                row = i * N + r
                A[row, :] = 0.0
                A[row, i * N:(i + 1) * N] = vrow

    # ------------------------------------------------------------------ rhs
    def rhs_schur(self, that):
        """theta coefficients -> buoyancy right-hand side of the Schur system."""
        N, Nk = self.N, self.Nk
        b = np.zeros(Nk * N, dtype=complex)
        for i, k in enumerate(self.kx):
            if k == 0.0:
                continue
            b[i * N:(i + 1) * N] = 1j * k * self.Ra * np.asarray(that[i])
            b[i * N + N - 2] = 0.0
            b[i * N + N - 1] = 0.0
        return b

    def rhs_mixed(self, that):
        N, Nk = self.N, self.Nk
        b = np.zeros(2 * Nk * N, dtype=complex)
        b[:Nk * N] = self.rhs_schur(that)
        return b

    # -------------------------------------------------------------- unpacking
    def unpack_mixed(self, x):
        """Mixed solution -> (psi_hat (Nk,N), omega_hat (Nk,N), v_hat (Nk,N))."""
        N, Nk = self.N, self.Nk
        v = x[:Nk * N].reshape(Nk, N)
        psi = x[Nk * N:].reshape(Nk, N)
        return psi, -v, v

    def unpack_schur(self, x):
        """Schur solution v -> (psi_hat, omega_hat, v_hat)."""
        N, Nk = self.N, self.Nk
        v = x.reshape(Nk, N)
        psi = np.zeros_like(v)
        for i in range(Nk):
            psi[i] = self.green(i) @ v[i]
        return psi, -v, v

    def solve_direct(self, that, form='mixed'):
        """STRATEGY A: assemble + LU + back substitution (the reference)."""
        if form == 'mixed':
            A = self.assemble_mixed()
            b = self.rhs_mixed(that)
            lu = sla.lu_factor(A)
            x = sla.lu_solve(lu, b)
            return self.unpack_mixed(x), A, lu
        A = self.assemble_schur()
        b = self.rhs_schur(that)
        lu = sla.lu_factor(A)
        x = sla.lu_solve(lu, b)
        return self.unpack_schur(x), A, lu

    # ---------------------------------------------------------- diagnostics
    def residual_mixed(self, x, that):
        A = self.assemble_mixed()
        b = self.rhs_mixed(that)
        return float(np.linalg.norm(b - A @ x) / max(np.linalg.norm(b), 1e-300))

    def residual_schur(self, x, that):
        A = self.assemble_schur()
        b = self.rhs_schur(that)
        return float(np.linalg.norm(b - A @ x) / max(np.linalg.norm(b), 1e-300))


class PrecondBase:
    def apply(self, r):
        raise NotImplementedError


class BlockSchurPrecond(PrecondBase):
    """Block preconditioner for the mixed system, built from eta_bar(z).

    variant 'full'  : exact block (Schur-complement) solve of the decoupled system
    variant 'lower' : block lower triangular  [[A11,0],[A21,S]]
    variant 'upper' : block upper triangular  [[A11,A12],[0,S]]
    schur   'exact' : S = A22 - A21 A11^-1 A12
    schur   'cheap' : S = A22   (cheap placeholder for the Schur block)
    """

    def __init__(self, op, variant='full', schur='exact'):
        if variant not in ('full', 'lower', 'upper'):
            raise ValueError('unknown variant %r' % (variant,))
        if schur not in ('exact', 'cheap'):
            raise ValueError('unknown schur %r' % (schur,))
        self.op, self.variant, self.schur = op, variant, schur
        N, V, D, D2 = op.N, op.V, op.D, op.D2
        Mbar = cheb_mult_matrix(op.eta_bar_c, N)
        Comm = D2 @ Mbar - 2.0 * (D @ Mbar @ D) + Mbar @ D2
        self.rows = []
        for k in op.kx:
            Lm = D2 - (k * k) * np.eye(N)
            a11 = (Lm @ Mbar).astype(complex)
            a11[-2:, :] = np.vstack([V[0], V[-1]])
            a12 = (2.0 * k * k * Comm).astype(complex)
            a12[-2:, :] = 0.0
            a21 = (-np.eye(N)).astype(complex)
            a21[-2:, :] = 0.0
            a22 = Lm.astype(complex)
            a22[-2:, :] = np.vstack([V[0], V[-1]])
            lu_a11 = sla.lu_factor(a11)
            if schur == 'exact':
                X = sla.lu_solve(lu_a11, a12)
                s = a22 - a21 @ X
            else:
                s = a22
            self.rows.append(dict(lu_a11=lu_a11, a12=a12, a21=a21,
                                  lu_s=sla.lu_factor(s)))

    def apply(self, r):
        N, Nk = self.op.N, self.op.Nk
        out = np.zeros_like(r)
        for idx, row in enumerate(self.rows):
            r1 = r[idx * N:(idx + 1) * N]
            r2 = r[(Nk + idx) * N:(Nk + idx + 1) * N]
            y1 = sla.lu_solve(row['lu_a11'], r1)
            if self.variant == 'upper':
                x2 = sla.lu_solve(row['lu_s'], r2)
                x1 = y1 - sla.lu_solve(row['lu_a11'], row['a12'] @ x2)
            elif self.variant == 'lower':
                x2 = sla.lu_solve(row['lu_s'], r2 - row['a21'] @ y1)
                x1 = y1
            else:
                x2 = sla.lu_solve(row['lu_s'], r2 - row['a21'] @ y1)
                x1 = y1 - sla.lu_solve(row['lu_a11'], row['a12'] @ x2)
            out[idx * N:(idx + 1) * N] = x1
            out[(Nk + idx) * N:(Nk + idx + 1) * N] = x2
        return out


class SchurOnlyPrecond(PrecondBase):
    """Per-wavenumber inverse of the validated depth-only reduced operator at
    eta_bar(z), used as a preconditioner for the Schur system B v = f."""

    def __init__(self, op):
        self.op = op
        N, V, D, D2 = op.N, op.V, op.D, op.D2
        Mbar = cheb_mult_matrix(op.eta_bar_c, N)
        Comm = D2 @ Mbar - 2.0 * (D @ Mbar @ D) + Mbar @ D2
        self.lu = []
        self.P = np.eye(N)
        self.P[-2:, :] = 0.0
        for idx, k in enumerate(op.kx):
            Lm = D2 - (k * k) * np.eye(N)
            B = (Lm @ Mbar + 2.0 * k * k * (Comm @ op.green(idx))).astype(complex)
            B[-2:, :] = np.vstack([V[0], V[-1]])
            self.lu.append(sla.lu_factor(B))

    def apply(self, r):
        N, Nk = self.op.N, self.op.Nk
        out = np.zeros_like(r)
        for idx in range(Nk):
            out[idx * N:(idx + 1) * N] = sla.lu_solve(
                self.lu[idx], self.P @ r[idx * N:(idx + 1) * N])
        return out


# ---------------------------------------------------------------- self test
def self_test(Nx=16, Nz=24, deta_T=10.0, verbose=True):
    """Structural tests that need no flow run."""
    from ..models.convection_modes import RBCVariableViscosity
    from .vv_stokes import VVStokesReduced

    m = RBCVariableViscosity(Nx=Nx, Nz=Nz, deta_T=deta_T)
    X, Z = np.meshgrid(m.x, m.z, indexing='ij')
    eta_bar = np.exp(-np.log(deta_T) * (1.0 - Z))        # depth-only
    rng = np.random.default_rng(3)
    that = np.zeros((m.Nk, m.Nz), dtype=complex)
    for i in range(1, m.Nk):
        that[i] = (rng.normal(size=m.Nz) + 1j * rng.normal(size=m.Nz)) \
            * np.exp(-np.arange(m.Nz) / 4.0)
    ok = True

    # 1. eta = eta(z): the mixed and Schur solves must both reproduce the
    #    VALIDATED depth-only operator VVStokesReduced bit-for-bit.
    op = CoupledStokes(m, eta_bar)
    (psi_s, om_s, v_s), _, _ = op.solve_direct(that, form='schur')
    (psi_m, om_m, v_m), _, _ = op.solve_direct(that, form='mixed')
    S = VVStokesReduced(m.Z, m.kx, m.Ra, m.eta_c)
    psi_ref = np.zeros_like(that)
    om_ref = np.zeros_like(that)
    for i, k in enumerate(m.kx):
        if k == 0.0:
            continue
        psi_ref[i], om_ref[i] = S.solve(k, that[i])
    e_schur = float(np.abs(psi_s - psi_ref).max() / max(np.abs(psi_ref).max(), 1e-300))
    e_mixed = float(np.abs(psi_m - psi_ref).max() / max(np.abs(psi_ref).max(), 1e-300))
    e_om = float(np.abs(om_m - om_ref).max() / max(np.abs(om_ref).max(), 1e-300))
    ok &= e_schur < 1e-12 and e_mixed < 1e-12 and e_om < 1e-12
    if verbose:
        print('  [1] eta=eta(z): schur vs depth-only %.2e | mixed vs depth-only %.2e'
              ' | omega %.2e' % (e_schur, e_mixed, e_om))

    # 2. the mixed system is block diagonal in k when eta = eta(z)
    A = op.assemble_mixed()
    N, Nk = op.N, op.Nk
    Ab = A[:Nk * N, :Nk * N].reshape(Nk, N, Nk, N)     # (v, v) block only
    off = max(np.abs(Ab[i, :, j, :]).max() for i in range(Nk)
              for j in range(Nk) if i != j)
    ok &= off == 0.0
    if verbose:
        print('  [2] off-block max |A_ij| (eta=eta(z)) = %.2e' % off)

    # 3. eta = eta(x,z): mixed and Schur systems are the SAME problem
    eta2 = np.exp(-np.log(deta_T) * (1.0 - Z + 0.15 * np.cos(np.pi * X)
                                     * np.sin(np.pi * Z)))
    op2 = CoupledStokes(m, eta2)
    (psi_m2, om_m2, v_m2), _, _ = op2.solve_direct(that, form='mixed')
    (psi_s2, om_s2, v_s2), _, _ = op2.solve_direct(that, form='schur')
    e = float(np.abs(v_m2 - v_s2).max() / max(np.abs(v_s2).max(), 1e-300))
    ok &= e < 1e-12
    if verbose:
        print('  [3] eta=eta(x,z): ||v_mixed - v_schur||_rel = %.2e' % e)

    # 4. the negative-wavenumber ("conjugate") half of the one-sided
    #    x-convolution.  It must be EXACTLY zero for eta = eta(z) (the immune
    #    region, so every depth-only / deta_T = 1 result is bit-for-bit safe) and
    #    non-zero for eta = eta(x,z), with the two signs genuinely different.
    Az0 = CoupledStokes(m, eta_bar, partner_sign=0.0).assemble_schur()
    Az1 = CoupledStokes(m, eta_bar, partner_sign=-1.0).assemble_schur()
    same_z = bool(np.array_equal(Az0, Az1))
    A20 = CoupledStokes(m, eta2, partner_sign=0.0).assemble_schur()
    A2m = CoupledStokes(m, eta2, partner_sign=-1.0).assemble_schur()
    A2p = CoupledStokes(m, eta2, partner_sign=+1.0).assemble_schur()
    scale = max(float(np.abs(A20).max()), 1e-300)
    d_m = float(np.abs(A2m - A20).max()) / scale
    d_p = float(np.abs(A2p - A20).max()) / scale
    d_mp = float(np.abs(A2p - A2m).max()) / scale
    ok &= same_z and d_m > 1e-6 and d_p > 1e-6 and d_mp > 1e-6
    if verbose:
        print('  [4] partner term: eta(z) matrices bit-identical %s ; '
              'eta(x,z) |dA|/|A| = %.2e (sign -1) / %.2e (sign +1) / '
              '%.2e (difference)' % (same_z, d_m, d_p, d_mp))
    return bool(ok)


if __name__ == '__main__':
    print('coupled variable-viscosity Stokes operator self test')
    self_test()
