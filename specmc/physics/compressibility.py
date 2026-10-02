"""Compressibility reference state: the TALA single source of truth.

Scope of this module
--------------------
TALA adds the Truncated Anelastic Liquid Approximation to the frozen
Boussinesq solver: the mass equation becomes ``div.(rho_bar u) = 0``, the
buoyancy is weighted by the reference density, and the energy equation carries
the reference-density mass weight, the adiabatic heating term and the
(reference-density weighted) radiogenic source.  All of that needs one thing
this package did not have before: a reference state.

The pressure-dependent viscosity module already put the algebra of King et al. (2010)'s profiles in
:mod:`specmc.physics.reference_state` (``rho_bar``/``T_bar``/``p_bar_king``) and
verified it against ``king_ala.prm:50`` to 2.22e-16.  This module is the
*contract* on top of it: the thing the equation layer consumes:

* one dissipation number ``Di``, and it is ``Config.dissipation_number``
  (the viscous-dissipation field).  Nothing here re-declares it: the frozen contract requires
  the same ``Di`` to drive the viscous-dissipation source ``(Di/Ra) Phi`` AND
  the compressibility,
  so a second field would break the single source of truth.  :func:`record`
  takes ``Di`` as an argument and never stores a copy;
* the Grueneisen parameter ``gamma`` (``king_ala.prm:46``), whose only job is
  the exponent ``Di/gamma``;
* the two derived objects the operator needs, and only those:

      a        = rho_bar'(z)/rho_bar(z) = -Di/gamma          (CONSTANT)
      rho_bar  = exp(-a * depth),  depth = 1 - z,  rho_bar(top) = 1

  ``a`` being constant is what lets the TALA streamfunction operator keep
  the project's per-wavenumber block structure (`specmc.stokes.stokes_tala`;
  the term list is confirmed by an independent longhand derivation).

The exact equations this state feeds (verbatim from the two reference
implementations):

    mass      div.(rho_bar u) = 0
    momentum  div.[eta (grad u + grad u^T - (2/3) div.u I)] - grad p'
                  - Ra rho_bar alpha_bar T' k_hat = 0
    energy    rho_bar cp_bar (dT'/dt + u.grad T') - div.[k_bar grad(T_bar+T')]
                  + Di alpha_bar rho_bar g.u T' - (Di/Ra) Phi = 0

    G-ADOPT `2d_compressible_TALA.py:61,65,75-77`;
    ASPECT manual sections 2.7.4/2.7.5 (`_lit/aspect_manual.txt:1626-1667`).

Provenance of every constant
----------------------------
======================  =========================  ==========================================
quantity                value                      source
======================  =========================  ==========================================
``Di``                  ``Config.dissipation_number`` (default 0.0)   CitcomS manual :685, :4080; ASPECT manual :24130
``gamma``               1.0                        ``king_ala.prm:46`` (``gamma=1.0``)
``T0 = T_surf/DeltaT``  0.091                      ``king_ala.prm:13``; ``reference_state.Ts_tilde``
``a = -Di/gamma``       derived                    ``rho_bar = exp(depth Di/gamma)``, ``king_ala.prm:50``
======================  =========================  ==========================================

Two conventions, and the one this project uses
----------------------------------------------
The published setups differ by a *constant offset* of the reference
temperature, and both are internally consistent:

* ASPECT/King, ``T_bar = T0 exp(depth Di)`` so ``T_bar(top) = T0``; the
  boundary temperatures are ``0.091`` (top) and ``1.091`` (bottom)
  (``king_ala.prm:113-123``);
* G-ADOPT, ``T_bar = T0 (exp(depth Di) - 1)`` so ``T_bar(top) = 0`` and
  the *total* temperature runs ``0 -> 1``
  (``2d_compressible_TALA.py:93,133``).

The perturbation ``T' = T - T_bar`` is the same field in both (the offsets
cancel), and this project's energy equation is written for
``T = (1 - z) + theta`` with homogeneous Dirichlet ``theta`` on both walls,
which is G-ADOPT's total-temperature convention.  :func:`T_bar_king` therefore
returns ASPECT's profile and :func:`T_bar_energy` G-ADOPT's; the difference is
``-T0`` and is documented rather than hidden.

What this module does NOT do
----------------------------
* it does not implement the ALA extra term ``(Di/gamma0)(cp0/cv0) rho_bar chi_T
  p'`` (TALA only);
* it does not support an arbitrary tabulated ``rho_bar(z)``: the equation layer
  consumes the constant ``a``.  The general-profile term list (which
  carries the ``rho_bar``-derivative factors) is a documented boundary of this
  round.

DOI
---
King, Lee, Van Keken, Leng, Zhong, Tan, Tosi & Kameyama (2010), GJI 180(1),
73-87, ``10.1111/j.1365-246X.2009.04403.x``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import reference_state as RS

__all__ = [
    "GAMMA_DEFAULT", "TALA_FORMULATIONS", "gamma_default", "T0_default",
    "a_coeff", "rho_bar_grid", "rho_bar", "T_bar_king", "T_bar_energy",
    "p_bar", "rho_bar_coeffs", "record", "describe", "self_test",
]

#: Grueneisen parameter of the King benchmark (``king_ala.prm:46``, the same
#: value the material model block repeats at ``:83``).
GAMMA_DEFAULT = 1.0

#: the two published reference-temperature normalisations (see the docstring)
TALA_FORMULATIONS = ("king", "energy")


def gamma_default():
    """-> the King benchmark's Grueneisen parameter (``king_ala.prm:46``)."""
    return GAMMA_DEFAULT


def T0_default(constants=None):
    """-> ``T0 = T_surf/DeltaT`` from the constants contract (``0.091``).

    This is ``reference_state.Ts_tilde``, the pV calibration's surface
    temperature and King's ``Adiabatic surface temperature`` (``king_ala.prm:13``)
    are the same number, which is why the compressibility contract
    reads it from there instead of hard-coding 0.091 a second time.
    """
    return RS.Ts_tilde(constants)


def _check(Di, gamma):
    Di = float(Di)
    gamma = float(gamma)
    if not np.isfinite(Di) or Di < 0.0:
        raise ValueError("Di must be finite and >= 0 (got %r)" % (Di,))
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be positive and finite (got %r)" % (gamma,))
    return Di, gamma


def a_coeff(Di, gamma=GAMMA_DEFAULT):
    """-> ``a = rho_bar'(z)/rho_bar(z) = -Di/gamma`` (CONSTANT).

    Sign: ``rho_bar = exp(depth Di/gamma)`` with ``depth = 1 - z`` increases
    with depth, so ``d rho_bar/dz = -a rho_bar`` with ``a = Di/gamma``; here the
    logarithmic derivative with respect to z, ``-(Di/gamma)``, is returned,
    because that is the coefficient that appears in the operator
    ``L = D^2 - a D - k^2`` (``stokes_tala.py``).
    ``Di = 0`` gives exactly ``-0.0`` and the operator collapses to the frozen
    Boussinesq one.
    """
    Di, gamma = _check(Di, gamma)
    return -Di / gamma


def rho_bar_grid(z, Di, gamma=GAMMA_DEFAULT):
    """-> ``rho_bar(z) = exp((1-z) Di/gamma)`` on the grid ``z``.

    ``rho_bar(top) = 1`` exactly (``z = 1`` => ``depth = 0`` => ``exp(0)``), the
    normalisation the King benchmark uses, and the one the operator's ``a``
    refers to.  ``Di = 0`` returns an array of exactly ``1.0``.
    """
    Di, gamma = _check(Di, gamma)
    if Di == 0.0:
        return np.ones_like(np.asarray(z, dtype=float))
    return RS.rho_bar(z, Di, gamma)


#: the name `reference_state` uses; kept as an alias so call sites can import the
#: profile from either module without a second implementation.
def rho_bar(z, Di, gamma=GAMMA_DEFAULT):
    """-> :func:`rho_bar_grid` (alias of the ``reference_state`` profile)."""
    return rho_bar_grid(z, Di, gamma)


def T_bar_king(z, Di, T0=None, constants=None):
    """-> ASPECT/King's reference temperature ``T0 exp((1-z) Di)``.

    ``king_ala.prm:50`` (first expression).  ``T_bar(top) = T0 = 0.091``.
    """
    if T0 is None:
        T0 = T0_default(constants)
    return RS.T_bar(z, Di, T0=float(T0))


def T_bar_energy(z, Di, T0=None, constants=None):
    """-> G-ADOPT's reference temperature ``T0 (exp((1-z) Di) - 1)``.

    ``2d_compressible_TALA.py:133``.  ``T_bar(top) = 0``, so the *total*
    temperature ``T_bar + theta`` carries the standard ``0 -> 1`` Dirichlet
    data with homogeneous ``theta``, this project's boundary convention.
    ``Di = 0`` gives exactly zero.
    """
    if T0 is None:
        T0 = T0_default(constants)
    return T_bar_king(z, Di, T0=T0, constants=constants) - float(T0)


def p_bar(z, Di, gamma=GAMMA_DEFAULT):
    """-> King's reference pressure ``gamma/Di (exp((1-z) Di/gamma) - 1)``.

    ``king_ala.prm:50`` (second expression).  It is the *anelastic* reference
    pressure and is NOT the pV pressure: the frozen pV contract uses the
    incompressible hydrostatic ``p0_norm(z) = 1 - z``
    (``reference_state``, the reference-hydrostatic pV contract).  Provided for
    the TALA interface/read-outs only; nothing in the equation layer consumes
    it.
    """
    Di, gamma = _check(Di, gamma)
    return RS.p_bar_king(z, Di, gamma)


def rho_bar_coeffs(Vi, z, Di, gamma=GAMMA_DEFAULT):
    """-> Chebyshev coefficients of ``rho_bar(z)`` for the tau representation.

    Helper for the equation layer, which needs the *coefficient* vector of the
    profile (the operators work in Chebyshev space).  ``Vi`` is the model's
    inverse Chebyshev transform (``RBC.Vi``), so the profile goes through
    exactly the same transform as every other z-field.  ``Di = 0`` returns an
    array with ``1.0`` in the first entry and exact zeros elsewhere, the
    constant field, as it must be for the frozen limit.
    """
    Di, gamma = _check(Di, gamma)
    z = np.asarray(z, dtype=float)
    prof = rho_bar_grid(z, Di, gamma)
    c = np.asarray(Vi @ prof, dtype=float)
    if Di == 0.0:
        c = np.zeros_like(c)
        c[0] = 1.0
    return c


def record(Di, gamma=GAMMA_DEFAULT, constants=None):
    """-> the validated compressibility record (JSON-safe) of one TALA run.

    Single entry point for the equation layer and the internal checks.  It
    never stores ``Di`` as a new parameter: ``Di = Config.dissipation_number``
    is passed in, and the record reports it back with the two derived
    objects and their provenance.
    """
    Di, gamma = _check(Di, gamma)
    T0 = T0_default(constants)
    a = a_coeff(Di, gamma)
    rec = dict(
        formulation="TALA",
        Di=float(Di),
        gamma=float(gamma),
        a=float(a),
        T0=float(T0),
        # the two profiles at the two walls, as exact closed forms
        rho_top=float(np.exp(0.0)),
        rho_bottom=float(np.exp(Di / gamma)),
        T_top_king=float(T0),
        T_bottom_king=float(T0 * np.exp(Di)),
        T_top_energy=0.0,
        T_bottom_energy=float(T0 * (np.exp(Di) - 1.0)),
        active=bool(Di != 0.0),
        rho_bar="exp((1-z) Di/gamma)",
        T_bar_king="T0 exp((1-z) Di)",
        T_bar_energy="T0 (exp((1-z) Di) - 1)",
        p_bar="(gamma/Di)(exp((1-z) Di/gamma) - 1)",
        source=("king_ala.prm:46,50,113-123; "
                "G-ADOPT 2d_compressible_TALA.py:61,65,75-77,93,133; "
                "ASPECT manual 1626-1667"),
        doi="10.1111/j.1365-246X.2009.04403.x",
    )
    return rec


def describe(Di, gamma=GAMMA_DEFAULT, constants=None):
    """-> a one-line summary (used by ``describe()`` dumps)."""
    r = record(Di, gamma, constants)
    if not r["active"]:
        return "boussinesq (Di = 0: rho_bar == 1, no reference state)"
    return ("TALA  Di=%g  gamma=%g  a=%.12g  rho_bar in [1, %.6g]  "
            "T0=%.4g" % (r["Di"], r["gamma"], r["a"], r["rho_bottom"],
                         r["T0"]))


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------
def self_test(verbose=True):
    """Structural checks of the compressibility contract.

    [1] ``Di = 0`` is the *exact* Boussinesq limit: ``rho_bar == 1``,
        ``T_bar_energy == 0``, ``p_bar == depth``, all componentwise, and
        ``a == 0.0``;
    [2] the two normalisations: ``rho_bar(top) = 1``, ``rho_bar(bottom) =
        exp(Di/gamma)``, ``T_bar_king(top) = T0``, ``T_bar_energy(top) = 0``,
        and ``T_bar_king - T_bar_energy == T0`` identically;
    [3] the King benchmark profiles reproduce ``king_ala.prm:50`` at Di = 1,
        gamma = 1 (the exact expressions, to 1e-15);
    [4] the logarithmic derivative: ``a`` is exactly ``rho_bar'/rho_bar`` and
        is CONSTANT in z (the property the operator relies on);
    [5] the SSOT link: ``T0`` is ``reference_state.Ts_tilde`` and the ``Di`` in
        the record is supplied by the calling code (never re-declared here);
    [6] unusable inputs raise.
    """
    ok = True

    # [1] the Di = 0 limit
    z = np.linspace(0.0, 1.0, 65)
    r0 = rho_bar_grid(z, 0.0)
    p0 = p_bar(z, 0.0)
    Te0 = T_bar_energy(z, 0.0)
    allone = bool(np.all(r0 == 1.0))
    allzero = bool(np.all(Te0 == 0.0))
    pdepth = float(np.abs(p0 - (1.0 - z)).max())
    ok &= allone and allzero and (a_coeff(0.0) == 0.0) and (pdepth == 0.0)
    if verbose:
        print("  [1] Di=0: rho_bar==1 %s, T_bar_energy==0 %s, p_bar==depth "
              "%.1e, a==0 %s"
              % (allone, allzero, pdepth, a_coeff(0.0) == 0.0))

    # [2] the two normalisations
    Di, gamma = 1.0, 1.0
    T0 = T0_default()
    top, bot = 1.0, 0.0
    ok &= float(rho_bar_grid(np.array([top]), Di, gamma)[0]) == 1.0
    ok &= abs(float(rho_bar_grid(np.array([bot]), Di, gamma)[0])
              - np.exp(Di / gamma)) < 1e-15
    ok &= abs(float(T_bar_king(np.array([top]), Di, T0)[0]) - T0) < 1e-16
    ok &= float(T_bar_energy(np.array([top]), Di, T0)[0]) == 0.0
    gap = float(np.abs(T_bar_king(z, Di, T0) - T_bar_energy(z, Di, T0)
                       - T0).max())
    ok &= gap < 1e-15
    if verbose:
        print("  [2] rho(top)=1, rho(bot)=e^{Di/gamma}=%.6f, T_bar_king(top)"
              "=T0=%.4g, T_bar_energy(top)=0, king-energy offset %.1e"
              % (np.exp(Di / gamma), T0, gap))

    # [3] king_ala.prm:50
    dep = np.array([0.0, 0.25, 0.5, 1.0])
    zb = 1.0 - dep
    e_rho = float(np.abs(rho_bar_grid(zb, Di, gamma)
                         - np.exp(dep * Di / gamma)).max())
    e_Tk = float(np.abs(T_bar_king(zb, Di, T0) - T0 * np.exp(dep * Di)).max())
    e_Te = float(np.abs(T_bar_energy(zb, Di, T0)
                        - (T0 * np.exp(dep * Di) - T0)).max())
    e_p = float(np.abs(p_bar(zb, Di, gamma)
                       - gamma / Di * (np.exp(dep * Di / gamma) - 1.0)).max())
    ok &= max(e_rho, e_Tk, e_Te, e_p) < 1e-15
    if verbose:
        print("  [3] king_ala.prm:50  rho %.1e  T_king %.1e  T_energy %.1e  "
              "p %.1e" % (e_rho, e_Tk, e_Te, e_p))

    # [4] a == rho_bar'/rho_bar, constant in z.  Checked through the EXACT
    #     multiplicative relation rho(z+h)/rho(z) = exp(-a h) (a finite
    #     difference of a smooth function would only be 2nd-order accurate and
    #     would test np.gradient, not the profile).
    worst = 0.0
    for D in (0.25, 0.5, 1.0):
        for h in (1e-3, 1e-2, 0.1):
            for zz in (0.0, 0.3, 0.7):
                if zz + h > 1.0:
                    continue
                ratio = float(rho_bar_grid(np.array([zz + h]), D, gamma)[0]
                              / rho_bar_grid(np.array([zz]), D, gamma)[0])
                # rho(z+h)/rho(z) = exp(a h)  =>  log(ratio)/h == a
                worst = max(worst, abs(np.log(ratio) / h - a_coeff(D, gamma)))
    ok &= worst < 1e-12
    if verbose:
        print("  [4] max |log(rho(z+h)/rho(z))/h - a| = %.1e (exact for "
              "Di in {0.25,0.5,1})" % worst)

    # [5] SSOT links
    ok &= (T0 == RS.Ts_tilde())
    rec = record(3.25, gamma=1.0)
    ok &= (rec["Di"] == 3.25) and (rec["a"] == -3.25) and rec["active"]
    ok &= (record(0.0)["active"] is False)
    if verbose:
        print("  [5] T0 == reference_state.Ts_tilde() == %.15g ; record(3.25)"
              "['a'] = %.15g" % (T0, rec["a"]))

    # [6] unusable inputs
    bad = 0
    for args in ((-1.0, 1.0), (float("nan"), 1.0), (1.0, 0.0), (1.0, -1.0),
                 (1.0, float("inf"))):
        try:
            record(*args)
        except ValueError:
            bad += 1
    ok &= (bad == 5)
    if verbose:
        print("  [6] unusable inputs rejected: %d/5" % bad)

    # [7] the Chebyshev helper the equation layer uses
    from ..core.rbc_galerkin import cgl, cheb_V
    N = 24
    zn = cgl(N)
    Vi = np.linalg.inv(cheb_V(zn, N))
    c0 = rho_bar_coeffs(Vi, zn, 0.0)
    zero_tail = bool(np.all(c0[1:] == 0.0))
    ok &= (c0[0] == 1.0) and zero_tail
    c1 = rho_bar_coeffs(Vi, zn, 1.0, 1.0)
    prof = cheb_V(zn, N) @ c1
    e = float(np.abs(prof - rho_bar_grid(zn, 1.0, 1.0)).max())
    ok &= e < 1e-13
    if verbose:
        print("  [7] rho_bar_coeffs: Di=0 -> [1, 0, ...] %s ; Di=1 round trip "
              "%.1e" % (zero_tail, e))
        print("  compressibility self_test %s"
              % ("PASSED" if ok else "FAILED"))
    return bool(ok)


if __name__ == "__main__":
    print("compressibility self test")
    self_test()
