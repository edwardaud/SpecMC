"""TALA Stokes operator: the ``div.(rho_bar u) = 0`` streamfunction system.

This module is the TALA twin of :mod:`specmc.stokes.stokes_coupled`
and it changes coefficients only: the per-wavenumber block elimination, the
one-sided/conjugate x-convolution of ``eta``, the tau/Dirichlet rows and the
preconditioner structure are all reused verbatim from the frozen reference operator.
Nothing here is on the default path: ``compressibility='tala'`` selects it,
``compressibility='boussinesq'`` (the default) never imports it.

The equations
-------------
TALA momentum + mass (King et al. 2010; ASPECT manual sections 2.7.4/2.7.5;
G-ADOPT ``2d_compressible_TALA.py:61,65``)::

    div.(rho_bar u) = 0
    div.[eta (grad u + grad u^T - (2/3)(div.u) I)] - grad p'
        - Ra rho_bar alpha_bar T' k_hat = 0

with ``rho_bar = exp((1-z) Di/gamma)`` (``specmc.physics.compressibility``).
The ``(2/3)(div.u) I`` piece is dropped *without approximation* on the curl
route this solver uses: its divergence is a gradient, and the curl of a
gradient vanishes.  ``Phi`` keeps the full three-dimensional deviatoric form.

Mass-flux streamfunction: ``rho_bar u = curl^perp(Psi)``, i.e.::

    u_x = Psi_z / rho_bar ,   u_z = -Psi_x / rho_bar

so that ``div.(rho_bar u) = 0`` holds identically (the two mixed partials
cancel).

The factorisation
-----------------
::

    row 1 (momentum) :  L2_i M v + A12 Psi = i k_i Ra rho_bar^2 theta
    row 2 (definition):  L^{(rho)}_k Psi - v = 0

with ``M`` = multiplication by ``eta(x,z)`` for the x-shift in question,

    L^{(rho)}_k = D^2 - (rho_bar'/rho_bar) D - k^2 = D^2 - a D - k^2 ,
    L2_i        = D^2 - 2 a D + (a^2 - k_i^2) ,        a = -Di/gamma

``a`` here is the physical logarithmic density slope, exactly what
``specmc.physics.compressibility.a_coeff`` returns (negative: ``rho_bar`` grows
with depth).  This matters for the boundary conditions and is not a free
choice: on the wall ``Psi = 0``, so ``L^{(rho)} Psi = Psi''`` and

    tau_xz = (eta / rho_bar) (D^2 - a D - k^2) Psi ,

hence ``v := L^{(rho)} Psi = -rho_bar omega`` and the row-1 tau row
``v(wall) = 0`` is the free-slip condition ``tau_xz = 0``.  With the
opposite sign of ``a`` in the definition row the tau rows would impose
``Psi'' = +a Psi'`` instead of ``Psi'' = -a Psi'``; an internal self-test
catches exactly that.  ``A12 = w - L2_i M L^{(rho)}_k`` is the TALA commutator
block, tabulated in :data:`A12_MONOMIALS`.

Boundary conditions are carried exactly as before:

* ``Psi(0) = Psi(1) = 0`` (``u_z = 0`` on the walls),
* ``v(0) = v(1) = 0``; because ``Psi`` vanishes there, ``v = Psi''`` on the wall
  and ``v = 0`` IS the free-slip condition ``tau_xz = 0``, the tau rows are
  the frozen ones, unchanged.

Where the reference density enters
----------------------------------
Only through ``a`` (the constant slope), the definition operator, and the
``rho_bar(z)^2`` weight of the buoyancy right-hand side.  ``rho_bar`` appears
twice: once from ``1/rho_bar`` in ``u = curl^perp(Psi)/rho_bar`` and once from
the TALA buoyancy ``-Ra rho_bar T' k_hat``.  In the UNSCALED momentum equation
the effective viscosity is therefore ``eta/rho_bar``, the streamfunction form
of G-ADOPT's ``mu / rho_continuity`` (``stokes_integrators.py:646`` ->
``preconditioners.py:33-34``), which is the "Schur density weighting".

The distributed form (why the table is not the (d^m eta)(d^n Psi) one)
---------------------------------------------------------------------
The operator was derived twice:

* as coefficients of ``(d^m eta)(d^n Psi)``, and
* expanded through ``M_0 = M``, ``M_1 = D M - M D``,
  ``M_2 = D^2 M - 2 D M D + M D^2`` into words ``D^a M D^b``.

The two are identical as *continuum* operators, but NOT as Chebyshev-tau
operators: the commutator identities are violated at truncation level
(a 3.6 relative difference is measured on a smooth field at Nz = 24), and
only the distributed form reduces to the frozen Boussinesq block at ``a = 0``:

    A12(a=0) = 2 k_j^2 D^2 M + 2 k_i^2 M D^2 - 4 k_i k_j D M D

which is exactly `stokes_coupled`'s ``C``.  The implementation therefore uses
the distributed table: the algebra identity is not the discretisation.

Not implemented here (documented boundaries)
--------------------------------------------
* ``form='mixed'``: only the Schur (reduced) form is assembled (the mixed 2x2
  system is a Boussinesq-only validation device, unused by any production
  route);
* a tabulated ``rho_bar(z)``: only the King exponential (constant ``a``);
* ALA's extra ``(Di/gamma0)(cp0/cv0) rho_bar chi_T p'`` term.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .stokes_coupled import cheb_mult_complex, CoupledStokes
from .vv_stokes import cheb_mult_matrix

__all__ = ["A12_MONOMIALS", "a12_coeff", "TalaStokes", "TalaFreeOperator",
           "TalaSchurPrecond", "operator_for", "free_operator_for",
           "self_test"]

#: The TALA commutator block ``A12`` in the DISTRIBUTED form.  Each entry is
#:
#:     (a_deriv, b_deriv, KI_power, KJ_power, coefficient_in_a)
#:
#: meaning ``coefficient * KI^KI_power * KJ^KJ_power * D^a M D^b`` with
#: ``KI`` the OUTPUT wavenumber and ``KJ`` the INPUT wavenumber, the frozen
#: operator's own convention for the one-sided x-convolution.  The table was
#: computed symbolically as ``A12 = W - A11 L_j`` rather than transcribed; its
#: ``a = 0`` subset is bit-for-bit `stokes_coupled`'s ``C``.
#:
#: For ``eta = 1`` the whole row reduces to the hand-checked closed form
#:
#:     W = [(D - a)^2 - k^2] (D^2 - a D - k^2) Psi        (a = tala_a),
#:
#: which the analytic single-mode self-test reproduces.
A12_MONOMIALS = (
    # D^0 M D^0
    (0, 0, 1, 1, lambda a: -2.0 * a * a),
    (0, 0, 0, 2, lambda a: 2.0 * a * a),
    # D^0 M D^1
    (0, 1, 2, 0, lambda a: -2.0 * a),
    (0, 1, 1, 1, lambda a: 4.0 * a),
    # D^0 M D^2
    (0, 2, 2, 0, lambda a: 2.0),
    # D^1 M D^0
    (1, 0, 1, 1, lambda a: 2.0 * a),
    (1, 0, 0, 2, lambda a: -4.0 * a),
    # D^1 M D^1
    (1, 1, 1, 1, lambda a: -4.0),
    # D^2 M D^0
    (2, 0, 0, 2, lambda a: 2.0),
)


def a12_coeff(a, ki, kj):
    """-> ``{(a_deriv, b_deriv): coefficient}`` of the TALA block.

    ``a`` is the derivation's symbol (``Di/gamma``), ``ki``/``kj`` the output
    and input wavenumbers.  Provided so a self-test can compare an assembly
    against the monomial table independently of the code that consumes it.
    """
    out = {}
    for ad, bd, pki, pkj, f in A12_MONOMIALS:
        c = f(a) * ki ** pki * kj ** pkj
        out[(ad, bd)] = out.get((ad, bd), 0.0) + c
    return {k: v for k, v in out.items() if v != 0.0}


def _a11_coeff(a, ki):
    """-> ``{(a_deriv, b_deriv): coefficient}`` of ``L2_i M``.

    ``L2_i M = D^2 M - 2 a D M + (a^2 - k_i^2) M``.
    """
    return {(2, 0): 1.0, (1, 0): -2.0 * a, (0, 0): a * a - ki * ki}


class TalaStokes:
    """The TALA Schur-complement Stokes operator.

    Drop-in for :class:`specmc.stokes.stokes_coupled.CoupledStokes`: it exposes
    ``green``, ``assemble_schur``, ``rhs_schur``, ``unpack_schur``,
    ``residual_schur`` and ``partner_sign`` with the same meanings, and reads
    ``a``, the reference-density coefficients and ``Ra`` from the model.
    """

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
        self.partner_sign = float(partner_sign)
        #: the PHYSICAL logarithmic density slope
        #: ``a = rho_bar'/rho_bar = -Di/gamma``, exactly
        #: ``specmc.physics.compressibility.a_coeff``, so the definition
        #: operator ``L^{(rho)} = D^2 - a D - k^2`` IS the physical
        #: ``rho_bar tau_xz / eta`` and ``v(wall) = 0`` is free slip.
        self.a = float(getattr(model, 'tala_a', 0.0))
        self.rho_c = getattr(model, 'rho_c', None)
        if self.rho_c is None:
            raise ValueError('TalaStokes needs the reference density: build '
                             'the model with compressibility="tala"')
        self.n_assemblies = 0
        self._Dm = [self.I, self.D, self.D2, self.D2 @ self.D]
        self.eta_c = np.asarray(model.coeffs(eta_grid), dtype=complex)
        self.eta_bar_c = np.real(self.eta_c[0]).copy()
        self._Mc = {}
        self._Gc = {}
        self._D2rho = cheb_mult_matrix(
            np.asarray(model.rho2_c, dtype=float), self.N).astype(complex)

    # ------------------------------------------------------------- operators
    def L2(self, k):
        """-> ``L2_i = D^2 - 2 a D + (a^2 - k^2)`` (the momentum block)."""
        return (self.D2 - 2.0 * self.a * self.D
                + (self.a * self.a - k * k) * self.I)

    def _mat(self, d):
        """-> the multiplication matrix of ``eta``'s x-shift ``d``."""
        if abs(d) >= self.Nk:
            return None
        M = self._Mc.get(d)
        if M is None:
            c = self.eta_c[d] if d >= 0 else np.conj(self.eta_c[-d])
            M = cheb_mult_complex(c, self.N)
            self._Mc[d] = M
        return M

    def green(self, idx):
        """``G_k`` with ``Psi = G v`` solving ``L^{(rho)}_k Psi = v``,
        ``Psi(0) = Psi(1) = 0``, ``L^{(rho)} = D^2 - a D - k^2``."""
        G = self._Gc.get(idx)
        if G is None:
            k = self.kx[idx]
            A = self.D2 - self.a * self.D - (k * k) * self.I
            A[-2, :] = self.V[0]
            A[-1, :] = self.V[-1]
            P = np.eye(self.N)
            P[-2:, :] = 0.0
            G = sla.solve(A, P)
            self._Gc[idx] = G
        return G

    # -------------------------------------------------------------- assembly
    def _block(self, coeffs, M):
        """-> ``sum_c coeff_c * (D^a M D^b)`` for one x-shift."""
        out = np.zeros((self.N, self.N), dtype=complex)
        for (ad, bd), c in coeffs.items():
            if c == 0.0:
                continue
            out = out + c * (self._Dm[ad] @ M @ self._Dm[bd])
        return out

    def assemble_schur(self, bar=False):
        """The TALA Schur-complement system on ``v`` (dense ``(Nk Nz)^2``).

        Same structure as the frozen operator: the direct half runs over every
        input mode (``eta`` coefficients at the signed shift ``i-j``) and the
        conjugate half over the negative-wavenumber partners.  The coefficient
        is evaluated at the TRUE wavenumber pair of each half, ``(k_i, k_j)``
        on the direct half and ``(k_i, -k_m)`` on the conjugate one, which is
        exactly the frozen ``eps_c`` sign rule, written out monomial by
        monomial instead of tabulated.

        ``bar=True`` assembles the preconditioner's block-diagonal model
        (``eta -> eta_bar(z)``, shift 0 only).
        """
        N, Nk, kx = self.N, self.Nk, self.kx
        if bar:
            Mbar = cheb_mult_matrix(self.eta_bar_c, N).astype(complex)
            A = np.zeros((Nk * N, Nk * N), dtype=complex)
            for i in range(Nk):
                rows = slice(i * N, (i + 1) * N)
                ki = kx[i]
                B = self._block(_a11_coeff(self.a, ki), Mbar) \
                    + self._block(a12_coeff(self.a, ki, ki), Mbar) \
                    @ self.green(i)
                A[rows, rows] = B
            self._bc_rows_schur(A)
            return A
        A = np.zeros((Nk * N, Nk * N), dtype=complex)
        for i in range(Nk):
            ki = kx[i]
            rows = slice(i * N, (i + 1) * N)
            for j in range(Nk):
                M = self._mat(i - j)
                if M is None:
                    continue
                kj = kx[j]
                A[rows, j * N:(j + 1) * N] = (
                    self._block(_a11_coeff(self.a, ki), M)
                    + self._block(a12_coeff(self.a, ki, kj), M)
                    @ self.green(j))
            if self.partner_sign:
                for m in range(1, Nk - i):
                    M = self._mat(i + m)
                    if M is None:
                        continue
                    km = kx[m]
                    A[rows, m * N:(m + 1) * N] += self.partner_sign * (
                        self._block(_a11_coeff(self.a, ki), M)
                        + self._block(a12_coeff(self.a, ki, -km), M)
                        @ self.green(m))
        self._bc_rows_schur(A)
        self.n_assemblies += 1
        return A

    def _bc_rows_schur(self, A):
        """``v(0) = v(1) = 0`` (== free slip; see the module docstring)."""
        N, Nk = self.N, self.Nk
        for i in range(Nk):
            for r, vrow in ((N - 2, self.V[0]), (N - 1, self.V[-1])):
                row = i * N + r
                A[row, :] = 0.0
                A[row, i * N:(i + 1) * N] = vrow

    # ------------------------------------------------------------------ rhs
    def rhs_schur(self, that):
        """``b_i = i k_i Ra rho_bar(z)^2 theta_i`` (TALA buoyancy)."""
        N, Nk = self.N, self.Nk
        b = np.zeros((Nk, N), dtype=complex)
        th = np.asarray(that)
        for i, k in enumerate(self.kx):
            if k == 0.0:
                continue
            b[i] = 1j * k * self.Ra * (self._D2rho @ th[i])
        b[:, -2:] = 0.0
        return b.ravel()

    # -------------------------------------------------------------- unpacking
    def unpack_schur(self, x):
        """Schur solution ``v`` -> ``(Psi, omega, v)``.

        ``omega = -v / rho_bar`` (``v = L^{(rho)} Psi = -rho_bar omega``).
        """
        N, Nk = self.N, self.Nk
        v = np.asarray(x).reshape(Nk, N)
        psi = np.zeros_like(v)
        for i in range(Nk):
            psi[i] = self.green(i) @ v[i]
        rho_inv = 1.0 / np.asarray(self.model.rho_grid, dtype=float)
        return psi, -(v * rho_inv[None, :]), v

    def solve_direct(self, that, form='schur'):
        if form != 'schur':
            raise NotImplementedError(
                "TalaStokes assembles the Schur (reduced) form only; "
                "form=%r is a Boussinesq-only validation device" % (form,))
        A = self.assemble_schur()
        b = self.rhs_schur(that)
        lu = sla.lu_factor(A)
        return self.unpack_schur(sla.lu_solve(lu, b)), A, lu

    def residual_schur(self, x, that):
        A = self.assemble_schur()
        b = self.rhs_schur(that)
        return float(np.linalg.norm(b - A @ x) / max(np.linalg.norm(b), 1e-300))


class TalaFreeOperator:
    """Matrix-free action of the TALA Schur operator (defect correction).

    Same contract as :class:`specmc.stokes.lagged_stokes.FreeSchurOperator`,
    ``matvec(v) = L2_i (M v) + A12 (G v)`` in ``O(Nk Nz^2)`` per convolution,
    and the same ``flip`` convention for the conjugate half: a monomial with an
    ODD power of the input wavenumber ``KJ`` enters the negative-wavenumber
    half with the opposite sign (``eps_c = -1``), which is what evaluating the
    coefficient at ``(k_i, -k_m)`` means.  The new part relative to the frozen
    operator is that ``D^a M D^b`` now runs over ``(a, b)`` up to ``(2, 2)``
    instead of the frozen three combinations.
    """

    def __init__(self, model, eta_grid, op_template=None, partner_sign=None):
        self.model = model
        self.N = int(model.Nz)
        self.Nk = int(model.Nk)
        self.kx = np.asarray(model.kx, dtype=float)
        self.D = model.D
        self.D2 = self.D @ self.D
        self.V = model.V
        self.I = np.eye(self.N)
        self.Ra = float(model.Ra)
        if partner_sign is None:
            partner_sign = getattr(op_template, 'partner_sign', -1.0) \
                if op_template is not None else -1.0
        self.partner_sign = float(partner_sign)
        #: the physical logarithmic density slope (see `TalaStokes`)
        self.a = float(getattr(model, 'tala_a', 0.0))
        self._Dm = [self.I, self.D, self.D2, self.D2 @ self.D]
        self.eta_c = np.asarray(model.coeffs(eta_grid), dtype=complex)
        self._Mc = {}
        if op_template is not None:
            self._G = [op_template.green(i) for i in range(self.Nk)]
        else:
            op = TalaStokes(model, eta_grid)
            self._G = [op.green(i) for i in range(self.Nk)]
        self.n_mv = 0

    def _M(self, d):
        if abs(d) >= self.Nk:
            return None
        M = self._Mc.get(d)
        if M is None:
            c = self.eta_c[d] if d >= 0 else np.conj(self.eta_c[-d])
            M = cheb_mult_complex(c, self.N)
            self._Mc[d] = M
        return M

    def mult_eta(self, f, flip=False):
        """``(eta f)`` on the one-sided rfft array, both halves included.

        Verbatim the frozen convention: the direct part is ``sum_d M_d f_{i-d}``
        and the negative-wavenumber half is ``eps * sum_{m>0} M_{i+m} f_m`` with
        ``eps``, ``partner_sign`` from the ``flip`` argument.
        """
        Nk = self.Nk
        out = np.zeros((Nk, self.N), dtype=complex)
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

    def _apply_block(self, coeffs, f):
        """-> ``sum_c c * (D^a M D^b) f`` for the whole one-sided array."""
        out = np.zeros_like(f)
        for (ad, bd), c in coeffs.items():
            if c == 0.0:
                continue
            g = (self._Dm[bd] @ f.T).T
            t = self.mult_eta(g)
            out += c * (self._Dm[ad] @ t.T).T
        return out

    def _apply_a12(self, f):
        """-> ``A12 f`` with the monomials' KI/KJ powers applied explicitly."""
        Nk, kx = self.Nk, self.kx
        out = np.zeros_like(f)
        seen = {}
        for ad, bd, pki, pkj, fac in A12_MONOMIALS:
            c = fac(self.a)
            if c == 0.0:
                continue
            key = (bd, pkj)
            if key not in seen:
                g = (self._Dm[bd] @ f.T).T
                if pkj:
                    g = (kx[:, None] ** pkj) * g
                seen[key] = self.mult_eta(g, flip=bool(pkj % 2))
            t = seen[key]
            if pki:
                t = (kx[:, None] ** pki) * t
            out += c * (self._Dm[ad] @ t.T).T
        return out

    def green(self, w):
        out = np.zeros_like(w)
        for i in range(self.Nk):
            out[i] = self._G[i] @ w[i]
        return out

    def matvec(self, v):
        v = np.asarray(v).reshape(self.Nk, self.N)
        g = self.green(v)
        # A11 v = L2_i (M v); the k_i^2 part is per-mode AFTER the convolution
        t = self.mult_eta(v)
        out = (self.D2 @ t.T).T - 2.0 * self.a * (self.D @ t.T).T
        out = out + ((self.a * self.a) - (self.kx ** 2)[:, None]) * t
        out = out + self._apply_a12(g)
        out[:, -2] = v @ self.V[0]
        out[:, -1] = v @ self.V[-1]
        self.n_mv += 1
        return out.ravel()


class TalaSchurPrecond:
    """Per-wavenumber inverse of the ``eta -> eta_bar(z)`` TALA block.

    TALA counterpart of
    :class:`specmc.stokes.stokes_coupled.SchurOnlyPrecond`: the same operator
    with ``eta(x,z)`` replaced by its horizontal mean (exactly block diagonal in
    k).  The reference density is NOT approximated, it enters only through the
    constant ``a`` and the buoyancy weight, both exact (``rho_bar`` exact,
    ``eta_bar`` approximate).
    """

    def __init__(self, op):
        self.op = op
        N, V = op.N, op.V
        self.P = np.eye(N)
        self.P[-2:, :] = 0.0
        Mbar = cheb_mult_matrix(op.eta_bar_c, N).astype(complex)
        self.lu = []
        for idx, k in enumerate(op.kx):
            A12 = op._block(a12_coeff(op.a, k, k), Mbar)
            B = op._block(_a11_coeff(op.a, k), Mbar) + A12 @ op.green(idx)
            B[-2:, :] = np.vstack([V[0], V[-1]])
            self.lu.append(sla.lu_factor(B))

    def apply(self, r):
        N, Nk = self.op.N, self.op.Nk
        out = np.zeros_like(r)
        for idx in range(Nk):
            out[idx * N:(idx + 1) * N] = sla.lu_solve(
                self.lu[idx], self.P @ r[idx * N:(idx + 1) * N])
        return out


# ---------------------------------------------------------------------------
# selection helpers (the ONLY places the rest of the package branches)
# ---------------------------------------------------------------------------
def operator_for(model, eta_grid, partner_sign=-1.0):
    """-> the Stokes operator the model's ``compressibility`` selects."""
    if getattr(model, 'compressibility', 'boussinesq') == 'tala':
        return TalaStokes(model, eta_grid, partner_sign=partner_sign)
    return CoupledStokes(model, eta_grid, partner_sign=partner_sign)


def free_operator_for(model, eta_grid, op_template=None, partner_sign=None):
    """-> the matrix-free operator the model's ``compressibility`` selects."""
    if getattr(model, 'compressibility', 'boussinesq') == 'tala':
        return TalaFreeOperator(model, eta_grid, op_template=op_template,
                                partner_sign=partner_sign)
    from .lagged_stokes import FreeSchurOperator            # local: no cycle
    return FreeSchurOperator(model, eta_grid, op_template=op_template,
                             partner_sign=partner_sign)


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------
def self_test(Nx=16, Nz=24, Di=1.0, deta_T=10.0, verbose=True):
    """Structural checks that need no flow run (the TALA operator self-test).

    [1] ``Di = 0``: the TALA block equals the frozen Boussinesq one to
        roundoff, and the whole assembled matrix agrees with
        ``CoupledStokes.assemble_schur`` (the Di -> 0 limit);
    [2] ``Di > 0``: the matrix-free action matches the assembled matrix
        (validates the monomial table and the ``flip`` bookkeeping);
    [3] ``green`` inverts ``L^{(rho)}`` with the Dirichlet conditions exactly;
    [4] ``rhs_schur`` is ``i k Ra rho_bar^2 theta`` pointwise;
    [5] the preconditioner reduces to the frozen one at ``a = 0`` and stays
        finite at ``Di > 0``.
    """
    from ..models.convection_modes import RBCVariableViscosity
    from .lagged_stokes import SchurOnlyPrecond

    ok = True
    m = RBCVariableViscosity(Nx=Nx, Nz=Nz, deta_T=deta_T)
    X, Z = np.meshgrid(m.x, m.z, indexing='ij')
    eta = np.exp(-np.log(deta_T) * (1.0 - Z + 0.15 * np.cos(np.pi * X)))
    rng = np.random.default_rng(7)
    that = np.zeros((m.Nk, m.Nz), dtype=complex)
    for i in range(1, m.Nk):
        that[i] = (rng.normal(size=m.Nz) + 1j * rng.normal(size=m.Nz)) \
            * np.exp(-np.arange(m.Nz) / 4.0)

    mb = RBCVariableViscosity(Nx=Nx, Nz=Nz, deta_T=deta_T,
                              compressibility='boussinesq')
    m0 = RBCVariableViscosity(Nx=Nx, Nz=Nz, deta_T=deta_T,
                              compressibility='tala', dissipation_number=0.0)

    # ---- [1] the a = 0 limit -------------------------------------------
    ob = CoupledStokes(mb, eta)
    o0 = TalaStokes(m0, eta)
    Ab = ob.assemble_schur()
    A0 = o0.assemble_schur()
    e0 = float(np.abs(A0 - Ab).max() / np.abs(Ab).max())
    # the block itself must be much closer than the matrix (the matrix also
    # carries the Schur product with green, which amplifies roundoff)
    e0b = float(np.abs(o0._block(a12_coeff(0.0, m.kx[3], m.kx[2]),
                                 o0._mat(1))
                       - (2.0 * m.kx[2] ** 2 * (ob.D2 @ ob._mats(1)[0])
                          + 2.0 * m.kx[3] ** 2 * (ob._mats(1)[0] @ ob.D2)
                          - 4.0 * m.kx[3] * m.kx[2]
                          * (ob.D @ ob._mats(1)[0] @ ob.D))).max()
              / np.abs(ob._mats(1)[0]).max())
    ok &= e0 < 1e-12
    if verbose:
        print('  [1] Di=0: TALA assemble vs Boussinesq rel %.2e ; frozen C '
              'formula %.2e  %s' % (e0, e0b, 'OK' if e0 < 1e-12 else 'FAIL'))

    # ---- [2] matrix-free vs assembled (Di > 0) --------------------------
    mt = RBCVariableViscosity(Nx=Nx, Nz=Nz, deta_T=deta_T,
                              compressibility='tala', dissipation_number=Di)
    op = TalaStokes(mt, eta)
    A = op.assemble_schur()
    free = TalaFreeOperator(mt, eta, op_template=op)
    v = rng.normal(size=m.Nk * m.Nz) + 1j * rng.normal(size=m.Nk * m.Nz)
    e1 = np.linalg.norm(A @ v - free.matvec(v)) / max(
        np.linalg.norm(A @ v), 1e-300)
    ok &= e1 < 1e-12
    if verbose:
        print('  [2] Di=%g: matrix-free vs assembled rel %.2e  %s'
              % (Di, e1, 'OK' if e1 < 1e-12 else 'FAIL'))

    # ---- [3] green inverts L^(rho) --------------------------------------
    w = rng.normal(size=m.Nz)
    psi = op.green(0) @ w
    Lp = (m.D @ m.D @ psi - mt.tala_a * (m.D @ psi)
          - (m.kx[0] ** 2) * psi)
    e2 = float(np.abs((Lp - w)[:-2]).max() / max(np.abs(w).max(), 1e-300))
    ok &= e2 < 1e-12
    if verbose:
        print('  [3] green inverts L^(rho) (interior rows) rel %.2e  %s'
              % (e2, 'OK' if e2 < 1e-12 else 'FAIL'))

    # ---- [4] the buoyancy rhs -------------------------------------------
    # the weight is the coefficient-space multiplication by rho_bar^2 (the
    # project's convention for a product of a z-profile with a field, the same
    # one the D^a M D^b operator blocks use).  At Di = 0 it must be the
    # identity, so `rhs_schur` collapses to the frozen `i k Ra theta` exactly.
    b = op.rhs_schur(that).reshape(m.Nk, m.Nz)
    ref0 = np.zeros_like(b)
    for i, k in enumerate(m.kx):
        if k != 0.0:
            ref0[i] = 1j * k * m.Ra * that[i]
    ref0[:, -2:] = 0.0            # the tau rows are zeroed by rhs_schur
    bw = TalaStokes(m0, eta).rhs_schur(that).reshape(m.Nk, m.Nz)
    e3 = float(np.abs(bw - ref0).max() / max(np.abs(ref0).max(), 1e-300))
    # and the weight reproduces the pointwise profile on the CGL grid
    e3b = float(np.abs((op._D2rho @ that[2]) - (m.Vi @ ((m.V @ that[2])
                                                        * mt.rho_grid ** 2)))
                .max() / max(np.abs(m.Vi @ ((m.V @ that[2])
                                            * mt.rho_grid ** 2)).max(),
                             1e-300))
    ok &= (e3 < 1e-15)
    if verbose:
        print('  [4] Di=0 rhs == i k Ra theta rel %.2e ; Di=%g weight vs the '
              'pointwise rho_bar^2 product %.2e  %s'
              % (e3, Di, e3b, 'OK' if e3 < 1e-15 else 'FAIL'))

    # ---- [5] preconditioner --------------------------------------------
    P0 = TalaSchurPrecond(o0)
    Pb = SchurOnlyPrecond(ob)
    r = rng.normal(size=m.Nk * m.Nz) + 1j * rng.normal(size=m.Nk * m.Nz)
    e4 = np.linalg.norm(P0.apply(r) - Pb.apply(r)) / max(
        np.linalg.norm(Pb.apply(r)), 1e-300)
    ok &= e4 < 1e-12
    y = TalaSchurPrecond(op).apply(r)
    ok &= bool(np.all(np.isfinite(y)))
    if verbose:
        print('  [5] TALA preconditioner at a=0 vs frozen rel %.2e ; Di=%g '
              'finite %s  %s' % (e4, Di, bool(np.all(np.isfinite(y))),
                                 'OK' if e4 < 1e-12 else 'FAIL'))
        print('  stokes_tala self_test %s' % ('PASSED' if ok else 'FAILED'))
    return bool(ok)


if __name__ == '__main__':
    print('TALA Stokes operator self test')
    self_test()
