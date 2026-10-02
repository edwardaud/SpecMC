"""Spectral BFBT preconditioner for the coupled (all-wavenumber) Stokes solve.

What "BFBT" is (and what this module is not)
--------------------------------------------
The name comes from Elman (1999), *Preconditioning for the steady-state
Navier-Stokes equations with low viscosity*, SIAM J. Sci. Comput. 20(4),
1299-1316, doi:10.1137/S1064827596312547: the *BFBt* preconditioner is

    S_BFBt^{-1} = (B B^T)^{-1} (B F B^T) (B B^T)^{-1},

with F the (1,1) momentum block itself, the trailing ``t`` is a transpose
marker, not "diagonal".  The ``D = diag(A)`` version is a *later* member of the
congruence family ``(B C^{-1} B^T)^{-1} (B C^{-1} A D^{-1} B^T) (B D^{-1} B^T)^{-1}``
(HyTeG's MantleConvection ``BFBTOperator`` / ``SchurBFBTSolver``; Rudi et al.
2017, arXiv:1607.03936), and ``diag(A)``-BFBT is known to be *mesh dependent*
while the weighted ``w``-BFBT (with ``M_u(mu^{-1/2})`` weights) is the variant
that is both mesh- and contrast-robust.  There is no public BFBT in CitcomS
(CitcomS-X is a 2009 AGU abstract with no released code); CitcomS's own pressure
preconditioner is a per-element diagonal of the Schur complement inside Uzawa.

This module implements the *spectral* translation, the target solver is not a
finite-element saddle-point system, so nothing is copied verbatim:

    finite element                     this solver
    ------------------------------     ------------------------------------
    D = diag(A) pointwise              D = the per-wavenumber (k-)block of the
                                       coupled Schur operator A(eta), inverted
                                       by one dense Nz x Nz LU per wavenumber
                                       (this is what "keep the wavenumber
                                       decoupling" means here)
    M_p = sparse pressure mass         M_p = the spectral Galerkin mass matrix
                                       V^T W V: the identity in the Fourier
                                       direction (so every operator stays
                                       k-block-diagonal) and dense Chebyshev in
                                       z.  For the CGL quadrature the
                                       mass-matrix projection coincides with
                                       the collocation transform, which
                                       `self_test` verifies.
    S_BFBT = B D^{-1} B^T              S_BFBT = the k-block-diagonal model of A,
                                       used as the preconditioner of the
                                       matrix-free FGMRES solve

Because ``eta`` enters ``A`` only through the horizontal convolution
``M_{i-j}``, the *literal* k-block-diagonal of ``A(eta)`` is not the
horizontal-mean operator: the conjugate ("partner") half of the one-sided
convolution contributes ``M_{i+i}`` to the diagonal block whenever
``2i < Nk``.  The ``'diag'`` profile below includes that term, so it is the
literal block-diagonal; ``'harm'`` and ``'geom'`` are the weighted variants
(`D^{-1}` = pointwise ``1/eta`` is the natural local inverse of the momentum
block, and its k-diagonal is the harmonic mean of ``eta``).

Everything here is a *preconditioner*: it changes how FGMRES converges, never
the solution.  `precond='legacy'` (the default) does not touch this module at
all, so the frozen reference production path stays bit for bit.
"""
import numpy as np
import scipy.linalg as sla

from .lagged_stokes import cheb_mult_matrix_fast
from .stokes_coupled import CoupledStokes

__all__ = ["PROFILES", "DEFAULT_PROFILE", "SpectralBFBTPrecond",
           "galerkin_mass", "galerkin_profile", "self_test"]

#: supported eta-profile choices for the k-block-diagonal model
PROFILES = ("diag", "harm", "geom")
#: the default: the pointwise-inverse (harmonic-mean) reading, see the module
#: docstring.  ``'diag'`` is the literal block-diagonal of ``A`` and is within
#: 1e-14 of ``'harm'`` when the partner term is dropped, which is why the
#: weighted form is the one that can differ from the legacy preconditioner.
DEFAULT_PROFILE = "harm"


def galerkin_mass(model):
    """-> the spectral Galerkin mass matrix ``M_p = V^T W V``.

    ``W = diag(wq)`` are the Clenshaw-Curtis weights of the CGL grid, so this is
    the mass matrix of the nodal basis pulled back to the Chebyshev-coefficient
    basis.  It is symmetric positive definite and dense in z; in x it is the
    identity, which is why the whole preconditioner stays k-block-diagonal.
    """
    V = np.asarray(model.V, dtype=float)
    w = np.asarray(model.wq, dtype=float)
    return V.T @ (w[:, None] * V)


def galerkin_profile(model, g):
    """-> ``M_p^{-1} V^T W g``: the Galerkin projection of a nodal profile.

    For an interpolatory quadrature this equals the collocation transform
    ``Vi @ g`` *for any* positive weight (the module self test checks the CGL
    case), so no new dense inverse is introduced by using M_p explicitly.
    """
    V = np.asarray(model.V, dtype=float)
    w = np.asarray(model.wq, dtype=float)
    return sla.solve(galerkin_mass(model), V.T @ (w * np.asarray(g, dtype=float)))


def _profile_nodal(eta_grid, profile):
    eta = np.asarray(eta_grid, dtype=float)
    if profile == "diag":
        return eta.mean(axis=0)
    if profile == "harm":
        return 1.0 / np.mean(1.0 / eta, axis=0)
    if profile == "geom":
        return np.exp(np.mean(np.log(eta), axis=0))
    raise ValueError("unknown profile %r (use one of %s)"
                     % (profile, ", ".join(PROFILES)))


class SpectralBFBTPrecond:
    """k-block-diagonal BFBT preconditioner for the coupled Schur system.

    Parameters
    ----------
    model : the convection model (needs ``Nk``, ``Nz``, ``kx``, ``V``, ``D``,
        ``wq``, ``green(i)`` and ``partner_sign``)
    eta_grid : (Nx, Nz) nodal viscosity field (the field the true operator uses)
    profile : one of :data:`PROFILES`
    include_partner : add the ``M_{2i}`` conjugate-half term to the diagonal
        block (``True`` = the *literal* block-diagonal of ``A(eta)``).  Measured
        on the production 64x96 state: the term is harmless for ``'diag'`` (716
        vs 695 iterations at 64x96, i.e. the literal block-diagonal IS
        the legacy operator) but destroys the weighted variants
        (``'harm'``/``'geom'`` go from 215/292 converged iterations to 800
        without converging), which is why the default is ``False``.
    """

    def __init__(self, model, eta_grid, profile=DEFAULT_PROFILE,
                 include_partner=False, op_template=None):
        self.m = model
        self.profile = str(profile)
        if self.profile not in PROFILES:
            raise ValueError("unknown profile %r (use one of %s)"
                             % (profile, ", ".join(PROFILES)))
        self.include_partner = bool(include_partner)
        self.N = int(model.Nz)
        self.Nk = int(model.Nk)
        self.kx = np.asarray(model.kx, dtype=float)
        self.D = model.D
        self.D2 = self.D @ self.D
        self.V = model.V
        self.I = np.eye(self.N)
        self.partner_sign = float(getattr(model, "partner_sign", -1.0))
        if op_template is not None:
            self._G = [op_template.green(i) for i in range(self.Nk)]
        else:
            op = CoupledStokes(model, eta_grid)
            self._G = [op.green(i) for i in range(self.Nk)]
        self.eta_c = model.coeffs(eta_grid)          # (Nk, N) complex
        g = _profile_nodal(eta_grid, self.profile)
        self.prof_c = np.real(galerkin_profile(model, g))
        self.n_partner_blocks = 0
        self._build()

    # -------------------------------------------------------------- assembly
    def _build(self):
        N, Nk, D2 = self.N, self.Nk, self.D2
        M0 = cheb_mult_matrix_fast(self.prof_c, N)
        Comm0 = D2 @ M0 - 2.0 * (self.D @ M0 @ self.D) + M0 @ D2
        self.lus = []
        for i in range(Nk):
            k = float(self.kx[i])
            k2 = k * k
            Lm = D2 - k2 * self.I
            P = (Lm @ M0 + 2.0 * k2 * (Comm0 @ self._G[i])).astype(complex)
            if self.include_partner and i > 0 and 2 * i < Nk:
                # the conjugate half of the one-sided x-convolution puts the
                # (i+i)-th horizontal mode on the diagonal block; this is what
                # makes the block-diagonal the LITERAL diagonal of A(eta)
                a2 = self.eta_c[2 * i]
                M2 = (cheb_mult_matrix_fast(np.real(a2), N)
                      + 1j * cheb_mult_matrix_fast(np.imag(a2), N))
                C2 = (2.0 * k2 * (D2 @ M2) + 2.0 * k2 * (M2 @ D2)
                      + 4.0 * k2 * (self.D @ M2 @ self.D))
                P += self.partner_sign * (Lm @ M2 + C2 @ self._G[i])
                self.n_partner_blocks += 1
            # tau rows of the Schur system: v(z=0) = v(z=1) = 0
            P[-2:, :] = np.vstack([self.V[0], self.V[-1]])
            self.lus.append(sla.lu_factor(P))
        self._P = np.eye(N)
        self._P[-2:, :] = 0.0

    # ------------------------------------------------------------------ apply
    def apply(self, r):
        N, Nk = self.N, self.Nk
        out = np.zeros_like(r)
        for i in range(Nk):
            out[i * N:(i + 1) * N] = sla.lu_solve(
                self.lus[i], self._P @ r[i * N:(i + 1) * N])
        return out

    # ----------------------------------------------------------- diagnostics
    def catalog(self):
        """-> a JSON-ready description of the constructed preconditioner."""
        return dict(kind="spectral-BFBT", profile=self.profile,
                    include_partner=self.include_partner,
                    n_wavenumbers=int(self.Nk),
                    n_partner_blocks=int(self.n_partner_blocks),
                    nz=int(self.N),
                    pressure_mass="V^T W V (Fourier identity x dense Chebyshev)")


# ---------------------------------------------------------------- self test
def self_test(Nx=16, Nz=24, verbose=True):
    """Structural checks that need no flow run.

    [1] the Galerkin mass matrix is SPD and its projection equals the
        collocation transform (so ``M_p`` introduces no new dense inverse);
    [2] at ``eta = const`` every profile gives the same preconditioner and it
        is the exact per-wavenumber inverse of the operator (FGMRES converges
        in one iteration);
    [3] the ``'diag'`` profile with and without the partner block differ for an
        ``eta(x, z)`` field (the partner term is real, not decoration);
    [4] every profile builds and applies without error, and ``apply`` is linear.
    """
    from ..models.convection_modes import RBCVariableViscosity
    from .lagged_stokes import LaggedStokes

    ok = True
    m = RBCVariableViscosity(Nx=Nx, Nz=Nz, deta_T=1000.0, symmetrize=True)
    X, Z = np.meshgrid(m.x, m.z, indexing='ij')
    eta_c = np.full_like(X, 1.0)
    Mp = galerkin_mass(m)
    spd = bool(np.all(np.linalg.eigvalsh(Mp) > 0.0))
    proj = galerkin_profile(m, eta_c.mean(axis=0))
    coll = m.Vi @ eta_c.mean(axis=0)
    e1 = float(np.abs(proj - coll).max() / max(np.abs(coll).max(), 1e-300))
    ok &= spd and e1 < 1e-12
    if verbose:
        print('  [1] M_p SPD %s ; Galerkin projection == collocation: %.2e'
              % (spd, e1))

    # [2] eta = const: the preconditioner is exact -> FGMRES in one iteration
    P = SpectralBFBTPrecond(m, eta_c, profile='diag')
    ls = LaggedStokes(m).refresh(eta_c)
    that = m.seed(amp=1e-2)
    x_d, _, _, _ = ls.solve_direct(that, eta_c)
    A = CoupledStokes(m, eta_c).assemble_schur()
    b = CoupledStokes(m, eta_c).rhs_schur(that)
    from ..core.krylov_fgmres import fgmres
    x_p, info, it, _ = fgmres(lambda v: A @ v, b, M=P.apply, rtol=1e-10,
                              restart=80, maxiter=200)
    e2 = float(np.linalg.norm(x_p - x_d) / max(np.linalg.norm(x_d), 1e-300))
    ok &= (it <= 2) and (e2 < 1e-8)
    if verbose:
        print('  [2] eta=1: preconditioned FGMRES iters %d, rel err %.2e'
              % (it, e2))

    # [3] eta(x,z): the partner term changes the diagonal blocks
    eta2 = np.exp(-np.log(1000.0) * (1.0 - Z + 0.15 * np.cos(np.pi * X)
                                     * np.sin(np.pi * Z)))
    Pa = SpectralBFBTPrecond(m, eta2, profile='diag', include_partner=False)
    Pb = SpectralBFBTPrecond(m, eta2, profile='diag', include_partner=True)
    eye = np.eye(m.Nz)
    dmax = max(float(np.abs(sla.lu_solve(Pa.lus[i], eye)
                            - sla.lu_solve(Pb.lus[i], eye)).max())
               for i in range(m.Nk))
    ok &= dmax > 0.0
    if verbose:
        print('  [3] partner block present in %d/%d wavenumbers; max inverse '
              'difference %.2e' % (Pb.n_partner_blocks, m.Nk, dmax))

    # [4] every profile applies, and apply is linear
    rng = np.random.default_rng(5)
    r = rng.normal(size=m.Nk * m.Nz) + 1j * rng.normal(size=m.Nk * m.Nz)
    for prof in PROFILES:
        P = SpectralBFBTPrecond(m, eta2, profile=prof)
        y = P.apply(r)
        lin = float(np.abs(P.apply(2.0 * r) - 2.0 * y).max())
        ok &= np.all(np.isfinite(y)) and lin < 1e-10
        if verbose:
            print('  [4] profile %-5s finite, linearity %.1e' % (prof, lin))

    if verbose:
        print('  bfbt self_test %s' % ('PASSED' if ok else 'FAILED'))
    return bool(ok)


if __name__ == '__main__':
    print('spectral BFBT preconditioner self test')
    self_test()
