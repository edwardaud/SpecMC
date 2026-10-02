"""Viscous dissipation for the 2D infinite-Prandtl convection model.

What this module is
===================
The single source of truth (SSOT) for the viscous-dissipation source
``Phi`` that viscous dissipation adds to the temperature equation of `specmc`.  With the
project's non-dimensionalisation (unit box height, unit thermal diffusivity,
``Delta T = 1``, unit thermal conductivity, ``z = 0`` the hot bottom) the
energy equation becomes

    d(theta)/dt + u.grad(theta) - u_z = lap(theta) + H + (Di/Ra) * Phi ,

    Phi = 2 eta ( eps - (1/3) tr(eps) 1 ) : eps
        = eta [ 4 psi_xz^2 + (psi_zz - psi_xx)^2 ] - (2/3) eta (div u)^2 ,

    u = (psi_z, -psi_x) ,   eps_ij = (u_i,j + u_j,i)/2 .

The second form is the one the code evaluates (the first is the definition);
on the project's stream-function representation ``div u == 0`` holds
identically, so the two agree to roundoff, see :func:`strain_fields`.

What this module does NOT do
============================
It does not know how the source is stepped in time.  The time discretisation is
the production one and is fixed by the frozen contract: the source is
explicit, it uses the Stokes velocity already frozen for the step (``u*``)
and the eta field that Stokes operator consumed, and it enters the
right-hand side only; ``matvec`` / ``precond`` / the tau rows /
``eta_pad_factor`` are untouched.
This module only evaluates ``Phi`` and its spectral image.

Coefficient: ``(Di/Ra)``, not 1
===============================
The dimensionless coefficient is ``Di/Ra``, and that is not a convention:

* CitcomS, manual, dimensionless energy equation, verbatim
  ``- rho_bar alpha g u_r Di (T+T_0) + (Di/Ra) Phi + rho_bar H``
  (``_lit/citcoms-manual.tex:635-640``; ``Di = alpha0 g0 R0/cP0`` at ``:685``),
  implemented at ``lib/Advection_diffusion.c:787`` (``temp = disptn_number /
  Atemp / vpts``) and ``:796`` (``heating = temp * visc * strain_sqr``);
* G-ADOPT, ``gadopt/approximations.py:400-402``, verbatim
  ``phi = inner(self.stress(u), grad(u)); return self.heating_weight * phi *
  self.Di / self.Ra`` (the adiabatic term on ``:404-406`` carries ``Di``, not
  ``Di/Ra``);
* ASPECT reaches the same coefficient through its non-dimensional material
  model ``eta = Di/Ra exp(...)``, ``alpha = Di``
  (``_lit/aspect_manual.txt:21748-21760``, ``Dissipation number`` default 0 at
  ``:24130``) with ``shear_heating.cc:57-65,87``.

A coefficient of ONE is numerically impossible in this project: the frozen
state has ``Nu = 10.037860428`` and the exact identity below gives
``<Phi> = Ra <theta u_z> = Ra (Nu_energy - 1) ~ 9.0e4``, so ``Di = 1`` would
drive ``d<T>/dt ~ 9e4``.  ``Di/Ra`` gives ``Di * 9.0``, a 0-100 % perturbation
of the ``Nu - 1 = 9.0`` transport, the physical size of the term.

The exact identity, and how it must be measured
===============================================
Contracting the momentum equation with ``u`` and integrating over the box (free
slip does no work, ``u.grad(p)`` integrates to zero) gives, for ANY state and
ANY ``eta(x,z)``,

    < Phi >  =  Ra < theta u_z > .                                    (*)

The identity is a statement about the *discrete* operator only if both sides are
evaluated by the SAME product pipeline.  The production pipeline for a source is
the ``eta``-dealiasing one of `specmc.models.coupled_model._eta_padded_truncated`
(``eta_pad_factor = 2`` zero-padded synthesis, pointwise evaluation, rfft
truncation back to ``Nk`` modes); :func:`pi_theta` evaluates the right-hand side
of (*) through that same pipeline.  Measured on the frozen 64x96 reference production
state (``stokes_mode='direct'``):

    ====================  ====================  ==============
    eta class             <Phi>                 rel. residual
    ====================  ====================  ==============
    eta = 1               3.892708388583998e+03  3.5e-16
    eta = eta(z)          4.456047428774324e+04  1.3e-13
    eta = eta(x,z)        9.038014912972425e+04  3.7e-11
    ====================  ====================  ==============

The residual grows with the x-spectrum of ``eta`` because the production
pipeline dealiases in ``x`` only: the ``z`` product aliases onto the resolved
Chebyshev modes, and ``Phi`` is a product of THREE fields while ``theta u_z`` is
a product of two.  The aliased evaluation of the right-hand side (``m.grid``
products, the plain pointwise convention) is 1e-6 .. 1e-5 away, so (*) is not a
test of the discretisation but of the PIPELINE unless both sides are treated
identically.  :data:`J9_2_TOL` records the measured, eta-class-dependent bound;
an earlier single ``max(1e-12, 10 tau_S)`` bound is superseded by the evidence
above (an erratum of the same kind as the earlier arithmetic corrections).

There is no z-dealiasing in production (`_numerics.DEALIAS_Z = False`, and
`implicit_theta` refuses it), so the bound below IS the achievable one; the
``Phi`` z-tail and the ``eta_pad_factor = 1`` negative control are measured
separately.

Energy budget
=============
Volume-integrating the temperature equation (incompressible, ``u.n = 0``,
insulating side walls) gives

    d<T>/dt = Nu_bot - Nu_top + H + (Di/Ra) <Phi>

so at a steady state ``Nu_top - Nu_bot = H + (Di/Ra) <Phi> >= 0``.  The source's
exact discrete volume integral is :func:`source_integral`; :func:`balance` is
the ``H + Phi`` generalisation of :func:`specmc.physics.heating.balance`.

Provenance of the form (four codes)
===================================
ASPECT
    manual equation (38), ``_lit/aspect_manual.txt:1441-1447``:
    ``+ 2 eta ( C eps(u) - (1/3) tr(C eps(u)) 1 ) : ( eps(u) - (1/3)(div u) 1 )``;
    ``source/heating_model/shear_heating.cc:57-63`` uses the DEVIATORIC strain
    rate when the model is compressible and the full one otherwise, ``:65``
    ``stress = 2 eta dev``, ``:87`` ``heating_source_terms = stress : dev``;
    registered verbatim at ``:236-243``.
CitcomS
    manual ``:635-640`` (``(Di/Ra) Phi``) and ``lib/Advection_diffusion.c:787,
    789,796``; the second invariant is built in
    ``lib/Viscosity_structures.c:1501-1545`` (``Vxyz[4..6] = 2 e12, 2 e13,
    2 e23``).
G-ADOPT
    ``gadopt/approximations.py:53-104`` (``deviatoric_tensor_from_grad``: sym,
    minus ``2/3 tr`` when compressible), ``:123`` (``stress_from_grad = mu *
    ...``), ``:400-402`` (``viscous_dissipation``).  The 2-D compressible TALA
    demo closes its own energy budget with exactly (*):
    ``_demos_mantle/2d_compressible_TALA/2d_compressible_TALA.py:237-238``
    ``energy_conservation_2 = abs(rate_work_against_gravity -
    rate_viscous_dissipation)``.
Underworld
    NEGATIVE control: ``AdvDiffusionSLCN`` has a source term ``f`` and no
    built-in dissipation (``advection_diffusion_eulerian.py:483-489,535``);
    ``strainrate_inv_II`` appears only in the plastic-yield viscosity
    (``constitutive_models.py:1494,1612-1617``).  Dissipation is a term a
    mature code must be asked for and must be validated globally.

King et al. (2010) ``10.1111/j.1365-246X.2009.04403.x`` tabulates ``Phi`` and
``W`` (``_extC_src/aspect/king_reference_statistics.txt:1``), which is (*) in a
compressible benchmark; its ``Phi ~ W`` agreement is 0.8 %-4 %, i.e. that table
can only judge an order of magnitude, and its non-dimensionalisation
(``eta = Di/Ra``, ``alpha = Di``) is not this project's.  Registered as an
TALA read-out, not as a viscous-dissipation check.
"""
from __future__ import annotations

import numpy as np

from . import heating as HEAT

__all__ = [
    "PARAMETER_MAPPING", "SECTION", "PHI_FORM", "DI_DEFINITION",
    "COEFFICIENT", "J9_2_TOL", "J9_2_TOL_GENERAL", "J9_2_TAIL_SLACK",
    "ETA_CLASSES",
    "eta_class", "strain_fields", "phi_padded", "phi_coeffs", "phi_field",
    "phi_integral", "phi_z_tail", "j9_2_bound", "pi_theta", "source_coeffs",
    "source_integral",
    "add_dissipation_source", "single_mode_theta_integral",
    "single_mode_psi_integral", "nu_top_steady", "nu_bot_steady", "balance",
    "describe", "reference_values", "self_test",
]

#: how the same term is spelled in the codes this form was chosen from
PARAMETER_MAPPING = {
    "ASPECT": dict(
        path="Heating model / Shear heating (+ 'Dissipation number')",
        units="-", term="2 eta (eps - (1/3) tr(eps) 1) : eps",
        note="source/heating_model/shear_heating.cc:57-65,87; the "
             "non-dimensional material model makes the term O(Di/Ra) "
             "(manual :21748-21760); 'shear heating' is not in the default "
             "heating list (manual :7272)"),
    "CitcomS": dict(
        path="input file keyword dissipation_number",
        units="-", term="(Di/Ra) Phi,  Di = alpha0 g0 R0/cP0",
        note="manual :635-640 (dimensionless energy equation, verbatim), "
             ":685 (Di definition), :4080 (default 0.0); "
             "lib/Advection_diffusion.c:787,789,796; the second invariant is "
             "lib/Viscosity_structures.c:1501-1545"),
    "G-ADOPT": dict(
        path="ExtendedBoussinesqApproximation.Ra / .Di (heating_weight)",
        units="-", term="phi * Di / Ra,  phi = inner(stress(u), grad(u))",
        note="gadopt/approximations.py:400-402; deviatoric stress at "
             ":53-104,123; the adiabatic sink at :404-406 carries Di alone"),
    "Underworld3": dict(
        path="(none)", units="-", term="no built-in viscous dissipation",
        note="AdvDiffusionSLCN has a source term f only "
             "(advection_diffusion_eulerian.py:483-489,535); "
             "strainrate_inv_II is plastic-yield only"),
}
SECTION = "viscous dissipation"

#: the definition, exactly as the two reference implementations state it
PHI_FORM = "2 eta (eps - (1/3) tr(eps) 1) : eps"
#: the dimensional meaning of the source field
DI_DEFINITION = "Di = alpha0 g0 d / cP0"
#: the dimensionless coefficient of Phi in the temperature equation
COEFFICIENT = "Di / Ra"

#: the three eta classes the identity's residual depends on, and the MEASURED
#: bound of |<Phi>/(Ra <theta u_z>) - 1| when both sides use the same
#: (production, x-dealiased) pipeline on the frozen 64x96 reference state:
#:   const -> 3.5e-16, eta_z -> 1.3e-13, eta_xz -> 3.7e-11 (direct LU)
#:   and 4.6e-11 through the production bfbt route.
J9_2_TOL = {
    "const": 1e-14,
    "eta_z": 1e-12,
    "eta_xz": 1e-9,
}
#: the bound used when the eta class is not known a priori
J9_2_TOL_GENERAL = 1e-9
ETA_CLASSES = ("const", "eta_z", "eta_xz")
#: The production pipeline has NO z-dealiasing, so the identity's residual is
#: the z-aliasing error of ``Phi``'s own pointwise products and is set by
#: ``Phi``'s z-Chebyshev tail.  Measured on the frozen 64x96 reference state
#: re-expanded on other z grids (measured):
#:
#:     Nz   <Phi>            rel. err    tail_z(Phi)   rel / tail
#:     24   9.187187519e+04  2.44e-02    5.10e-02      4.8e-01   (unresolved)
#:     48   9.038122975e+04  6.70e-06    5.30e-04      1.3e-02
#:     72   9.038015010e+04  6.24e-09    1.07e-05      5.8e-04
#:     96   9.038014913e+04  3.79e-11    1.79e-07      2.1e-04
#:
#: The rule is therefore state-dependent: the residual must stay below
#: ``J9_2_TAIL_SLACK`` times ``Phi``'s OWN unresolved z-content, which is what
#: :func:`j9_2_bound` returns.  At the production resolution it is inactive
#: (the class bound dominates); at a coarser grid it correctly declares the
#: state under-resolved instead of failing a converged run.
J9_2_TAIL_SLACK = 0.05

#: candidate divergence-free Gauss-Legendre node count for the analytic checks
_GL_N = 96


# ---------------------------------------------------------------------------
# the padded pipeline (the SAME one `eta_of` uses; nothing new is invented)
# ---------------------------------------------------------------------------
def _pad_factor(model):
    """-> the x-dealiasing factor the model's ``eta`` pipeline uses.

    ``1`` is the aliased (pointwise) negative control; every production model
    uses ``eta_pad_factor = 2`` (``_numerics.ETA_DEALIAS``), and a model without
    that attribute, the isoviscous ``RBC`` of the explicit route, uses the
    3/2-rule factor 2 whenever ``model.dealias`` is on.  The default
    configuration pins both values, so ``Phi`` and ``eta`` always share one
    padding width.
    """
    fac = int(getattr(model, "eta_pad_factor", 0) or 0)
    if fac == 0:
        fac = 2 if bool(getattr(model, "dealias", False)) else 1
    return fac


def _G(model, cf):
    """Chebyshev coeffs -> grid values, through the PRODUCTION eta pipeline.

    ``eta_pad_factor >= 2`` calls ``model._synth_pad``, the
    zero-padded synthesis of `coupled_model._eta_padded_truncated`; the
    ``eta_pad_factor = 1`` negative control falls back to the plain
    ``model.grid`` (the aliased pointwise path).
    """
    if _pad_factor(model) <= 1:
        return model.grid(cf)
    return model._synth_pad(cf)


def _F(model, g):
    """(padded) grid values -> Chebyshev coeffs, through the same pipeline.

    This is `ImplicitThetaSolver._F` / the truncation step of
    `_eta_padded_truncated`; it is repeated here as a module-level function of
    the MODEL because the dissipation source is evaluated before the theta
    solver exists, and because the physics module must not import
    `specmc.core`.
    """
    if _pad_factor(model) <= 1:
        A = np.fft.rfft(g, axis=0) / model.Nx
    else:
        A = model._trunc_pad(g)
    return (model.Vi @ A.T).T


def _eta_grid(model, eta):
    """Accept eta as a z-profile, an (Nx,Nz) grid, or None (= 1)."""
    if eta is None:
        return None
    e = np.asarray(eta, dtype=float)
    if e.ndim == 1:
        e = np.tile(e[None, :], (int(model.Nx), 1))
    if e.shape != (int(model.Nx), int(model.Nz)):
        raise ValueError("eta must be (Nz,) or (Nx,Nz) = (%d,%d) (got %r)"
                         % (model.Nx, model.Nz, np.shape(eta)))
    return e


def eta_class(eta):
    """-> 'const' | 'eta_z' | 'eta_xz' for a (possibly None) eta field.

    The classification is exact (not a tolerance): the project's eta fields are
    either uniform, a function of z alone, or a full ``eta(x,z)``.
    """
    if eta is None:
        return "const"
    e = np.asarray(eta, dtype=float)
    if e.ndim == 1 or e.size == 0:
        return "eta_z"
    if np.all(e == e.flat[0]):
        return "const"
    if np.all(e == e[0:1, :]):
        return "eta_z"
    return "eta_xz"


# ---------------------------------------------------------------------------
# the strain-rate fields (exact stream-function algebra)
# ---------------------------------------------------------------------------
def strain_fields(model, psi):
    """-> the strain-rate components of ``u = (psi_z, -psi_x)``, as Chebyshev
    coefficient fields of shape ``(Nk, Nz)``:

        eps_xx =  psi_xz ,   eps_zz = -psi_xz ,   eps_xz = (psi_zz - psi_xx)/2

    and ``tr(eps) = eps_xx + eps_zz``, which is identically zero for a stream
    function velocity field (the two terms are the same array with opposite
    signs, up to the order of two floating-point operations).  Returned as a
    tuple ``(exx, ezz, exz, tr)``.
    """
    k = model.kx[:, None]
    D = model.D
    psi = np.asarray(psi)
    cxz = (1j * k) * (D @ psi.T).T          # psi_xz
    czz = (D @ (D @ psi.T)).T               # psi_zz
    cxx = -(k ** 2) * psi                   # psi_xx
    exx = cxz
    ezz = -cxz
    exz = 0.5 * (czz - cxx)
    return exx, ezz, exz, exx + ezz


def phi_padded(model, psi, eta=None, synth=None):
    """-> ``Phi`` on the padded ``(2 Nx, Nz)`` grid, without the tau rows.

    ``Phi = 2 eta (eps - (1/3) tr(eps) 1) : eps`` evaluated pointwise, with
    every factor synthesised by the production ``eta_pad_factor`` pipeline.  The
    ``(2/3) eta (div u)^2`` correction is carried explicitly:
    it is ~1e-32 on this project's stream-function fields, and it is exactly
    what a compressible (TALA) implementation will need.

    ``synth`` overrides the synthesis operator (default :func:`_G`, the
    production ``eta_pad_factor`` one).  The only other call site is the
    z-dealiased reference path ``RBCVariableViscosity._nonlinear_coeffs_z``,
    which passes its own ``_Gz`` so that the source lands on the same grid that
    the nonlinearity is assembled on.
    """
    G = _G if synth is None else synth
    exx, ezz, exz, tr = strain_fields(model, psi)
    gxx = G(model, exx)
    gzz = G(model, ezz)
    gxz = G(model, exz)
    gtr = G(model, tr)
    sq = gxx ** 2 + gzz ** 2 + 2.0 * gxz ** 2
    dev = sq - (1.0 / 3.0) * gtr ** 2
    if eta is None:
        return 2.0 * dev
    ep = _eta_grid(model, eta)
    return 2.0 * G(model, model.coeffs(ep)) * dev


def phi_coeffs(model, psi, eta=None, tau_zero=True):
    """-> the ``(Nk, Nz)`` Chebyshev-per-x-mode spectral image of ``Phi``.

    ``tau_zero=True`` (the production value) zeroes the two tau rows, because
    those rows of the right-hand side are the (zero) Dirichlet boundary rows,
    not equation rows, exactly the convention of
    :func:`specmc.physics.heating.add_constant_source` and of
    ``ImplicitThetaSolver.rhs``.
    """
    c = _F(model, phi_padded(model, psi, eta))
    if tau_zero:
        c[:, -2:] = 0.0
    return c


def phi_field(model, psi, eta=None):
    """-> ``Phi`` resynthesised on the physical ``(Nx, Nz)`` grid."""
    return model.grid(phi_coeffs(model, psi, eta))


def phi_integral(model, psi, eta=None):
    """-> the discrete volume integral ``<Phi>`` of the source that is added.

    It is the integral of the TRUNCATED field, the same object that enters the
    right-hand side, so it is the quantity the discrete energy budget of
    :func:`balance` must close with.
    """
    return float(model.vol_mean(phi_field(model, psi, eta)))


def j9_2_bound(eta_cls='eta_xz', tau_S=0.0, phi_tail=0.0):
    """-> the state-dependent bound of the identity (*).

    ``max(class bound, 10 tau_S, J9_2_TAIL_SLACK * tail_z(Phi))``.  The first
    term is the measured resolved-state floor (see :data:`J9_2_TOL`), the second
    covers a Stokes solve that is only converged to its own residual, and the
    third the z-aliasing error that the un-dealiased ``Phi`` product carries on
    an under-resolved state (see :data:`J9_2_TAIL_SLACK`).
    """
    return max(J9_2_TOL.get(str(eta_cls), J9_2_TOL_GENERAL),
               10.0 * float(tau_S),
               J9_2_TAIL_SLACK * float(phi_tail))


def phi_z_tail(model, psi, eta=None, n_last=6):
    """-> ``Phi``'s relative z-Chebyshev tail (worst x-mode / global max).

    This is the quantity the third term of :func:`j9_2_bound` needs; it must be
    computed WITHOUT zeroing the tau rows, otherwise the zero rows hide the
    tail.
    """
    c = np.abs(phi_coeffs(model, psi, eta, tau_zero=False))
    g = max(float(c.max()), 1e-300)
    return float(c[:, -n_last:].max() / g)


def pi_theta(model, psi, that, dealias=True):
    """-> ``Ra <theta u_z>`` through the SAME pipeline as :func:`phi_integral`.

    ``dealias=True`` (the default, and the only value the identity (*) may be
    judged with) synthesises ``u_z`` and ``theta`` on the padded grid, forms the
    pointwise product and truncates it back to ``Nk`` modes; ``dealias=False``
    reproduces the plain ``m.grid`` product (1e-6 .. 1e-5 away from (*) on a
    developed state, which is why it is not the default).
    """
    that = np.asarray(that)
    psi = np.asarray(psi)
    if not dealias:
        uz = model.grid(-1j * model.kx[:, None] * psi)
        th = np.real(model.grid(that))
        return float(model.Ra * model.vol_mean(uz * th))
    uz = _G(model, -1j * model.kx[:, None] * psi)
    th = _G(model, that)
    cf = _F(model, uz * th)
    return float(model.Ra * model.vol_mean(model.grid(cf)))


# ---------------------------------------------------------------------------
# the source
# ---------------------------------------------------------------------------
def source_coeffs(model, psi, eta=None, Di=0.0, Ra=None, tau_zero=True):
    """-> ``(Di/Ra) * Phi_hat`` as an ``(Nk, Nz)`` coefficient array.

    ``Di = 0`` (the frozen reference value) returns an all-zero array WITHOUT
    evaluating the law, and :func:`add_dissipation_source` then returns its
    input object unchanged: the bit-for-bit immune path.
    """
    out = np.zeros((int(model.Nk), int(model.Nz)), dtype=complex)
    if float(Di) == 0.0:
        return out
    Ra_v = float(model.Ra if Ra is None else Ra)
    if not np.isfinite(Ra_v) or Ra_v <= 0.0:
        raise ValueError("Ra must be finite and positive (got %r)" % (Ra_v,))
    out += (float(Di) / Ra_v) * phi_coeffs(model, psi, eta, tau_zero=tau_zero)
    return out


def source_integral(model, coeffs):
    """-> the exact discrete volume integral of a source coefficient array.

    ``sum_j wq_j f(z_j)`` for a Chebyshev coefficient vector is ``gk . c``
    (``gk_0 = 1``, ``gk_1 = 0``, ``gk_2 = -1/3``, ...), and the x-average of a
    coefficient field keeps the ``k_x = 0`` mode only; the implementation
    uses the model's own quadrature, so no weight is re-typed here.
    """
    c = np.asarray(coeffs)
    return float(model.vol_mean(model.grid(c)))


def add_dissipation_source(b, coeffs, Di):
    """-> ``b`` with the dissipation source added to the EQUATION rows.

    ``b`` is the raveled ``(Nk, Nz)`` right-hand side of the theta equation and
    its last two columns are the Dirichlet rows (already zeroed by the calling
    code).
    ``Di == 0`` returns the same object, so the frozen reference code path is
    bit-for-bit what it was (the ``H = 0`` rule of
    :func:`specmc.physics.heating.add_constant_source`).
    """
    if float(Di) == 0.0:
        return b
    c = np.asarray(coeffs)
    out = np.array(b, copy=True).reshape(c.shape)
    out[:, :-2] += c[:, :-2]
    return out.ravel()


# ---------------------------------------------------------------------------
# analytic reference (eta = 1, single mode)
# ---------------------------------------------------------------------------
def single_mode_theta_integral(Ra, kx, kz, amp):
    """-> ``<Phi>`` for ``theta = amp cos(kx x) sin(kz z)`` and ``eta = 1``.

    With ``gamma = kx^2 + kz^2`` the Stokes solution is
    ``psi = A sin(kx x) sin(kz z)``, ``A = -Ra kx/gamma^2 amp``, and

        <Phi> = Ra^2 kx^2 / (4 gamma^2) * amp^2 = A^2 gamma^2 / 4

    (measured: ``6.3325739776e5`` for ``Ra = 1e4``, ``kx = kz = pi``,
    ``amp = 1``).
    """
    g = float(kx) ** 2 + float(kz) ** 2
    return float(Ra) ** 2 * float(kx) ** 2 / (4.0 * g ** 2) * float(amp) ** 2


def single_mode_psi_integral(kx, kz, A):
    """-> ``<Phi>`` for a PRESCRIBED stream function ``A sin(kx x) sin(kz z)``.

    Equal to ``A^2 gamma^2 / 4``; the two single-mode helpers differ only in
    which amplitude is taken as given (``A = -Ra kx/gamma^2 amp_theta``).
    """
    g = float(kx) ** 2 + float(kz) ** 2
    return float(A) ** 2 * g ** 2 / 4.0


# ---------------------------------------------------------------------------
# energy bookkeeping
# ---------------------------------------------------------------------------
def nu_top_steady(uz_theta, H=0.0, phi_total=0.0):
    """-> ``Nu_top`` at a steady state with both sources on.

    ``Nu_top = 1 + <u_z theta> + H/2 + (Di/Ra)<Phi>/2``: whatever is produced
    inside must leave through the top, so the H-only plateau of
    :func:`specmc.physics.heating.nu_top_conductive` shifts by HALF the
    dissipation source's volume integral as well.
    """
    return 1.0 + float(uz_theta) + 0.5 * float(H) + 0.5 * float(phi_total)


def nu_bot_steady(uz_theta, H=0.0, phi_total=0.0):
    """-> ``Nu_bot`` at the same steady state: ``Nu_top - H - (Di/Ra)<Phi>``."""
    return (1.0 + float(uz_theta) - 0.5 * float(H)
            - 0.5 * float(phi_total))


def balance(model, that_np1, that_n, that_nm1, a0, b0, b1, H=0.0, Di=0.0,
            psi=None, eta=None, advection=None, Ra=None):
    """-> the five terms (and residual) of the ``H + Phi`` energy budget.

    Identical to :func:`specmc.physics.heating.balance`, which is CALLED, not
    re-implemented, so the ``H``-only path is bit-for-bit unchanged, plus

        dissipation = (Di/Ra) <Phi> ,   residual -= dissipation .

    ``psi``/``eta`` default to the state's own Stokes solution and to the eta
    the model last consumed (``last_eta``, else ``eta_grid``, else 1), which is
    exactly the pair the source used.
    """
    base = HEAT.balance(model, that_np1, that_n, that_nm1, a0, b0, b1, H=H,
                        advection=advection)
    if float(Di) == 0.0:
        base["dissipation"] = 0.0
        return base
    if psi is None:
        psi, _ = model.stokes(that_np1)
    if eta is None:
        eta = getattr(model, "last_eta", None)
        if eta is None:
            eta = getattr(model, "eta_grid", None)
    Ra_v = float(model.Ra if Ra is None else Ra)
    diss = (float(Di) / Ra_v) * phi_integral(model, psi, eta)
    out = dict(base)
    out["dissipation"] = float(diss)
    out["residual"] = float(base["residual"]) - float(diss)
    return out


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def describe(Di=0.0, Ra=1e4):
    """-> a JSON-safe description of the active dissipation setting."""
    Di = float(Di)
    Ra = float(Ra)
    return dict(section=SECTION, dissipation_number=Di, Ra=Ra,
                enabled=bool(Di != 0.0),
                coefficient=COEFFICIENT,
                coefficient_value=(Di / Ra if Di != 0.0 else 0.0),
                phi_form=PHI_FORM, di_definition=DI_DEFINITION,
                equation="d(theta)/dt + u.grad(theta) - u_z = lap(theta) + H "
                         "+ (Di/Ra) Phi",
                time_discretisation="explicit: u* = the step's frozen Stokes "
                                    "velocity, eta = the field the Stokes "
                                    "operator consumed",
                dealiasing="eta_pad_factor padded synthesis -> pointwise Phi "
                           "-> rfft truncation to Nk modes (no z-dealiasing)",
                parameter_mapping=dict(PARAMETER_MAPPING),
                j9_2_tolerance=dict(J9_2_TOL), j9_2_general=J9_2_TOL_GENERAL,
                j9_2_bound_rule="max(class bound, 10*tau_S, %.2f*tail_z(Phi))"
                                % J9_2_TAIL_SLACK)


def reference_values(Di=0.0, Ra=1e4, kx=None, kz=None, amp=1.0):
    """-> the analytic single-mode reference values for ``Di``/``Ra``."""
    kx = np.pi if kx is None else float(kx)
    kz = np.pi if kz is None else float(kz)
    phi = single_mode_theta_integral(Ra, kx, kz, amp)
    return dict(Di=float(Di), Ra=float(Ra), kx=kx, kz=kz, amp=float(amp),
                phi_integral=phi,
                source_integral=float(Di) / float(Ra) * phi,
                coefficient_value=float(Di) / float(Ra))


# ---------------------------------------------------------------------------
# self test (runs in seconds; wired into the package self-tests)
# ---------------------------------------------------------------------------
def self_test(Nx=32, Nz=48, verbose=True):
    """Structural checks of the module's algebra (no march, no drivers)."""
    ok = True
    from ..models.convection_modes import make_convection

    m = make_convection(Ra=1e4, Nx=Nx, Nz=Nz)
    Ra = float(m.Ra)
    X, Z = np.meshgrid(m.x, m.z, indexing='ij')

    # [1] analytic single mode through the full padded pipeline (eta = 1):
    #     psi = A sin(kx x) sin(kz z)  ->  <Phi> = A^2 gamma^2 / 4
    e1 = 0.0
    for (mm, nn) in ((1, 1), (1, 2), (2, 1)):
        kx, kz = mm * np.pi, nn * np.pi
        A = 3.7e-2
        psi = m.coeffs(A * np.sin(kx * X) * np.sin(kz * Z))
        got = phi_integral(m, psi, None)
        want = single_mode_psi_integral(kx, kz, A)
        e1 = max(e1, abs(got - want) / abs(want))
    ok &= e1 < 1e-12
    if verbose:
        print('  [1] analytic single mode (eta=1): max rel err %.2e (<=1e-12)'
              % (e1,))

    # [2] the identity (*) is exact to roundoff when BOTH sides use the padded
    #     pipeline on an exact single-mode Stokes solution (eta = 1)
    kx = kz = np.pi
    that = m.coeffs(1e-3 * np.cos(kx * X) * np.sin(kz * Z))
    psi, _ = m.stokes(that)
    lhs = phi_integral(m, psi, None)
    rhs = pi_theta(m, psi, that, dealias=True)
    rhs_plain = pi_theta(m, psi, that, dealias=False)
    e2 = abs(lhs - rhs) / abs(rhs)
    ok &= e2 < 1e-13
    if verbose:
        print('  [2] identity on one exact mode: <Phi> = %.15e, Ra<th u_z> = '
              '%.15e, rel %.2e (plain %.2e)' % (lhs, rhs, e2,
                                                abs(lhs - rhs_plain)
                                                / abs(rhs_plain)))

    # [3] Di = 0 is a hard no-op: the source array is all-zero and
    #     add_dissipation_source returns the SAME object
    b = (np.arange(float(m.Nk * m.Nz)).reshape(m.Nk, m.Nz).astype(complex)
         + 0.5j)
    c0 = source_coeffs(m, psi, None, Di=0.0)
    b0 = add_dissipation_source(b.ravel(), c0, 0.0)
    same = (b0 is b.ravel()) or np.shares_memory(b0, b)
    ok &= (float(np.abs(c0).max()) == 0.0) and same
    if verbose:
        print('  [3] Di=0 no-op: |coeffs|=%.1e, same object=%s'
              % (float(np.abs(c0).max()), same))

    # [4] Di != 0 touches the EQUATION rows only: the two tau columns of the
    #     right-hand side are untouched, and the delta is (Di/Ra) Phi exactly.
    #     (b is O(1) so that bD - b is a roundoff-free difference.)
    Di = 0.5
    b = (np.full((m.Nk, m.Nz), 1e-3, dtype=complex) + 1e-4j)
    cD = source_coeffs(m, psi, None, Di=Di)
    bD = add_dissipation_source(b.ravel(), cD, Di).reshape(b.shape)
    d = bD - b
    tau_ok = float(np.abs(d[:, -2:]).max()) == 0.0
    scaled = float(np.abs(d[:, :-2]
                          - (Di / Ra) * phi_coeffs(m, psi, None)[:, :-2]).max())
    ok &= tau_ok and scaled <= 1e-18
    if verbose:
        print('  [4] Di=%.3g: tau rows untouched=%s, delta == (Di/Ra)Phi to %.1e'
              % (Di, tau_ok, scaled))

    # [5] the divergence-free correction: 2 eta (eps - tr/3)^2 equals
    #     eta[4 psi_xz^2 + (psi_zz-psi_xx)^2] to roundoff on this project's
    #     stream-function fields, and div u is zero to roundoff
    exx, ezz, exz, tr = strain_fields(m, psi)
    dtr = float(np.abs(tr).max() / max(np.abs(exx).max(), 1e-300))
    plain = (4.0 * _G(m, exx) ** 2
             + (2.0 * _G(m, exz)) ** 2)
    dev = phi_padded(m, psi, None)
    e5 = float(np.abs(dev - plain).max() / max(np.abs(plain).max(), 1e-300))
    ok &= (dtr < 1e-14) and (e5 < 1e-14)
    if verbose:
        print('  [5] div u == 0 to %.1e; deviatoric == plain to %.1e'
              % (dtr, e5))

    # [6] source_integral of the source is (Di/Ra) times <Phi>, and the
    #     analytic single-mode value of reference_values is consistent
    si = source_integral(m, cD)
    ok &= abs(si - (Di / Ra) * phi_integral(m, psi, None)) <= 1e-12 * abs(si)
    ref = reference_values(Di=Di, Ra=Ra)
    ok &= abs(ref['source_integral']
              - (Di / Ra) * ref['phi_integral']) == 0.0
    if verbose:
        print('  [6] source_integral %.6e == (Di/Ra)<Phi>; reference_values '
              'consistent' % (si,))

    # [7] the eta classification is exact and describe() is JSON-safe
    ok &= (eta_class(None) == 'const'
           and eta_class(np.ones(Nz)) == 'eta_z'
           and eta_class(np.ones((Nx, Nz))) == 'const'
           and eta_class(np.tile(np.linspace(1, 2, Nz), (Nx, 1))) == 'eta_z'
           and eta_class(X * 0.0 + np.linspace(1, 2, Nz)[None, :]
                         + 0.5 * X) == 'eta_xz')
    d = describe(Di=Di, Ra=Ra)
    ok &= (d['coefficient'] == COEFFICIENT and d['enabled']
           and abs(d['coefficient_value'] - Di / Ra) == 0.0)
    if verbose:
        print('  [7] eta_class exact; describe(Di=%g) coefficient=%r value=%r'
              % (Di, d['coefficient'], d['coefficient_value']))

    # [8] the bound rule is the max of its three terms, and the Phi z-tail
    #     is a real, positive number on a solved state
    b8 = j9_2_bound('eta_xz', tau_S=1e-8, phi_tail=1e-3)
    ok &= (b8 == max(J9_2_TOL['eta_xz'], 1e-7, J9_2_TAIL_SLACK * 1e-3)
           and j9_2_bound('const', 0.0, 0.0) == J9_2_TOL['const'])
    tl = phi_z_tail(m, psi, None)
    ok &= (np.isfinite(tl) and tl >= 0.0)
    if verbose:
        print('  [8] identity bound = max(class, 10 tau_S, %.2f tail) = %.3e; '
              'Phi z-tail on the test state %.3e'
              % (J9_2_TAIL_SLACK, b8, tl))

    if verbose:
        print('  dissipation self_test %s' % ('PASSED' if ok else 'FAILED'))
    return bool(ok)


if __name__ == '__main__':
    print('viscous dissipation self test')
    self_test()
