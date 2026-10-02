"""Arrhenius (temperature-only) viscosity law: single source of truth.

What this module is
-------------------
The frozen reference rheology is the exponential (Frank-Kamenetskii *linearised*)
law of :func:`specmc.physics.viscosity.eta_exp_T`::

    eta(T) = exp( -ln(Delta eta_T) * T ),      T in [0, 1],   D = Delta eta_T

which is a straight line in ``ln eta`` versus ``T``.  The physical law is the
Arrhenius (activated-creep) form, in which ``ln eta`` is linear in ``1/T``
instead::

    eta = eta_0 * exp( (E_a + p V_a) / (R T) )

This module owns the T-only member of that family (``p V_a`` is NOT part of
it yet; see :data:`PV_CONTRACT` and ``specmc/config.py``).

The calibration (the project's own normalisation)
-------------------------------------------------
The project writes ``T = 1 - z + theta`` with ``z = 0`` the hot bottom and
``z = 1`` the cold top, and defines the reference viscosity ``eta == 1`` at the
coldest material (that is what fixes ``Ra``).  The exponential law is
calibrated by the two endpoint values

    eta(0) = 1                (cold end, stiff)
    eta(1) = 1 / D            (hot end, weak)

and the Arrhenius family member that satisfies *both* endpoints exactly is
::

    ln eta(T) = E / (T + Ts) + C
    E = ln(D) * Ts * (1 + Ts)
    C = -E / Ts = -ln(D) * (1 + Ts)

i.e. :func:`eta_arrhenius_T` computes
``eta = exp( E/(T+Ts) - E/Ts )``.  ``Ts`` (= ``arrhenius_Ts``) is the only free
parameter; it is a *temperature offset* in the same units as ``T``.  For every
``Ts > 0`` the law is smooth, strictly decreasing in ``T``, and reproduces the
two endpoints exactly (to the roundoff of the algebra below).

Equivalent anchored form (used by the internal checks, not by the law)
----------------------------------------------------------------------
The same expression can be written without ``ln D``::

    ln eta(T) = E * ( 1/(T+Ts) - 1/Ts )

and the usual CitcomS/Cookbook5 parameterisation
``eta = eta0 exp( viscE/(T + viscT) - viscE/(1 + viscT) )`` is recovered with
``viscE = E``, ``viscT = Ts``.  That form has the *hot* end (``T = 1``) equal
to ``eta0``; here the normalisation is fixed at the *cold* end, so the mapping
carries an overall factor, see :data:`CITCOMS_MAPPING`.

Relationship to the frozen law, and the quantitative target
-----------------------------------------------------------
``Ts -> inf`` gives ``E ~ ln(D) * Ts**2`` and ``ln eta -> -ln(D) T`` exactly, so
the exponential law is the ``Ts -> inf`` limit of this family.  At any finite
``Ts`` the two share the endpoints but differ in between; the family member
closest to the frozen law has ``sup|eta_arrhenius - eta_exp| = 0.128`` for
``D = 1000`` (measured).  The slope at the cold end
is amplifed by

    |d ln eta/dT|(T=0) = E / Ts**2 = ln(D) * (1 + Ts)/Ts

relative to the constant ``ln D`` of the exponential law: ``Ts = 1 -> 2x``,
``0.5 -> 3x``, ``0.2 -> 6x``, ``0.1 -> 11x``.  ``Ts = 1`` gives ``E = 2 ln D``,
which is exactly the Frank-Kamenetskii activation used by
``frank_kamenetskii.cc`` (``E_FK = 2 ln(Delta eta_T)``) and recorded in
``specmc.physics.viscosity``.

Provenance
----------
* CitcomS ``lib/Viscosity_structures.c`` (``RHEOL == 3``)::

      TT[kk] = max(TT[kk], zero);
      temp += min(TT[kk], one) * ...
      EEta = tempa * exp( E[l]/(temp + T[l]) - E[l]/(one + T[l]) );

  ``T`` normalised to ``[0, 1]``, ``viscE = E``, ``viscT = T_0``, the same
  functional form, clamped to the calibration interval at assembly time.
  ``RHEOL`` defaults to 3.  Cookbook5 (the official Arrhenius example) states
  ``viscE = 24``, ``viscT = 0.182`` and enforces ``visc_max/visc_min =
  100/0.01``; its *physical* contrast ``exp(E/((1+T0)(2+T0))) = exp(10.16)``
  is ``2.6e10``, which the window compresses to ``1e4``.
* ASPECT ``source/material_model/rheology/frank_kamenetskii.cc``::
  ``viscosity = prefactor * exp(viscosity_ratio * 0.5 * (1 - T/reference_temperature))``
  is the linearisation of this law about ``T = T_ref``; matching the endpoint
  ratio gives ``viscosity_ratio = E_FK = 2 ln(Delta eta_T)``, and its
  ``reference_temperature`` defaults to the *adiabatic surface temperature*.
* ASPECT ``source/material_model/rheology/diffusion_dislocation.cc`` implements
  the full physical form ``exp((E + p V)/(n R T))`` with ``std::clamp`` in the
  first layer of every creep law (default window ``1e17/1e28`` Pa s).  That
  ``p V`` term is the subject of :data:`PV_CONTRACT` and is not implemented.

What this module does NOT do
-----------------------------------------
* No dynamic/full pressure term: the pressure is the *reference hydrostatic*
  one and nothing else.  A dynamic-pressure coupling is a different field with
  a different numerical cost (an outer eta-p fixed point); see
  :data:`PV_CONTRACT`.
* No depth term of the CitcomS ``viscZ``/ASPECT ``depth prefactor`` kind beyond
  the reference-hydrostatic ``p0(z)`` term defined by the frozen contract.
* No grain size, stress, or non-Newtonian dependence.
* No new bound/clip: the window of ``specmc.physics.eta_bounds`` is applied
  downstream in the padded space, exactly as for the exponential law.

The pV law is implemented
---------------------------------------------
``arrhenius_V > 0`` selects the reference-hydrostatic pV law of the frozen
contract,

    ln eta(T, z) = (E-tilde + V-tilde * p0_norm(z)) / (T + Ts) + C ,
    p0_norm(z) = 1 - z   (from :mod:`specmc.physics.reference_state`, the SSOT)

evaluated by :func:`arrhenius_exponent_V` in the cancellation-free two-term
form

    ln eta = -E T / (Ts (T + Ts)) + V p0_norm / (T + Ts) .

``E-tilde``, ``V-tilde``, ``Ts`` and ``C`` come from the ONE calibration
procedure :func:`specmc.physics.reference_state.calibrate_pv`.  The
``arrhenius_V = 0`` path does not go through this function: it takes the
Arrhenius T-only branch bit for bit, which is why the old
``arrhenius_exponent`` above is untouched.
"""
import numpy as np

__all__ = [
    "arrhenius_E", "arrhenius_C", "arrhenius_exponent", "eta_arrhenius_T",
    "slope_amplification", "cold_end_slope", "oracle_eta", "oracle_ln_eta",
    "infer_deta_T", "infer_Ts", "reference_check", "self_test",
    "arrhenius_exponent_V", "eta_arrhenius_V", "hot_end_contrast_V",
    "PV_CONTRACT", "CITCOMS_MAPPING",
]


# ---------------------------------------------------------------------------
# the pV contract (defined and frozen; now implemented)
# ---------------------------------------------------------------------------
#: The pV contract for the ``arrhenius_V`` field, frozen in the Arrhenius law and
#: implemented by the pressure-dependent viscosity law.
#:
#: ``arrhenius_V`` is the dimensionless coefficient of the reference
#: (hydrostatic) pressure term of the full Arrhenius exponent,
#:
#:     V-tilde = rho0 * g * d * V_a / (R * Delta T)
#:
#: (``rho0`` reference density, ``g`` gravity, ``d`` layer depth, ``V_a``
#: activation volume, ``R`` gas constant, ``Delta T`` the temperature drop
#: across the layer), multiplying the reference hydrostatic pressure
#: ``P = p0(z) = rho0 g d (1 - z)``, i.e. the reference-hydrostatic pV
#: evaluation.  The default ``0.0`` means "pressure term off"; the T-only law
#: of this module is then exact and bit-for-bit reproducible.
#:
#: Scope limits (frozen with the Arrhenius law and unchanged by the implementation):
#:
#: * the pressure is the reference hydrostatic pressure, never the dynamic
#:   pressure the Stokes solve produces.  A dynamic/full-pressure coupling is a
#:   *different field* with a different numerical cost (an outer eta-p fixed
#:   point), and is explicitly outside ``arrhenius_V``'s meaning;
#: * ``p0`` comes from the reference-state single source of truth
#:   (:mod:`specmc.physics.reference_state`); no viscosity module re-types
#:   ``rho g z``;
#: * the two endpoints do not determine ``(E, V)`` once ``V != 0``: a third
#:   physical anchor is required.  ``Ts`` and ``V-tilde`` come from the
#:   dimensional constants, and exactly one of ``E-tilde`` / the contrast ``D``
#:   is the third anchor (:func:`specmc.physics.reference_state.calibrate_pv`);
#: * ``arrhenius_V = 0`` remains a bit-for-bit no-op of the T-only law; the
#:   pV branch in ``RBCCoupled._eta_from_T`` is entered before any float
#:   arithmetic only when ``arrhenius_V > 0``.
#:
#: The physically required magnitude is
#: ``p V_a / E_a = 1.06`` (diffusion) to ``2.47`` (dislocation/Peierls) for
#: ASPECT's default activation volume, i.e. ``O(1)``.
PV_CONTRACT = dict(
    field="arrhenius_V",
    symbol="V-tilde",
    definition="rho0 * g * d * V_a / (R * Delta T)",
    pressure="reference hydrostatic p0(z) = rho0 g d (1 - z)",
    default=0.0,
    implemented=True,
    implemented_by="the pressure-dependent viscosity law",
    calibration="specmc.physics.reference_state.calibrate_pv",
    constants="specmc.physics.reference_state.ReferenceConstants",
    excluded_scope=("dynamic pressure coupling", "full (total) pressure",
                    "TALA/ALA reference-state expansion"),
    law="pressure-dependent viscosity",
)

#: Mapping of this module's calibration onto the published sources.  ``viscE``
#: / ``viscT`` are CitcomS's names; ``E_FK`` is ASPECT's Frank-Kamenetskii
#: activation (the ``Ts = 1`` member).
CITCOMS_MAPPING = dict(
    D="exp(viscE / ((1 + viscT) * ... )), see CITCOMS_MAPPING note",
    note=("CitcomS normalises at the HOT end: eta = N0 exp(viscE/(T+viscT) - "
          "viscE/(1+viscT)).  This module normalises at the COLD end "
          "(eta(0) = 1).  The two agree with N0 = exp(-viscE * (1 + 2*viscT) / "
          "((1 + viscT) * viscT)) at viscE = E, viscT = Ts."),
    viscE="E = ln(D) * Ts * (1 + Ts)",
    viscT="Ts",
    E_FK="E at Ts = 1 = 2 ln(D)",
)


# ---------------------------------------------------------------------------
# the law
# ---------------------------------------------------------------------------
def arrhenius_E(deta_T, Ts):
    """-> the activation ``E = ln(Delta eta_T) * Ts * (1 + Ts)``.

    This is the value that makes :func:`arrhenius_exponent` satisfy
    ``eta(0) = 1`` and ``eta(1) = 1 / deta_T`` simultaneously.  At ``Ts = 1``
    it is ``2 ln(deta_T)``, ASPECT's Frank-Kamenetskii activation ``E_FK``.
    """
    D = float(deta_T)
    s = float(Ts)
    if not (np.isfinite(D) and D > 0.0):
        raise ValueError("deta_T must be positive and finite (got %r)"
                         % (deta_T,))
    if not (np.isfinite(s) and s > 0.0):
        raise ValueError("arrhenius_Ts must be positive and finite (got %r)"
                         % (Ts,))
    return float(np.log(D) * s * (1.0 + s))


def arrhenius_C(deta_T, Ts):
    """-> the additive constant ``C = -E / Ts = -ln(Delta eta_T) * (1 + Ts)``."""
    return -arrhenius_E(deta_T, Ts) / float(Ts)


def arrhenius_exponent(T, deta_T, Ts):
    """-> the Arrhenius exponent ``ln(eta)``, anchored and cancellation-free.

    Starting point (the calibrated law, exact at both endpoints by
    construction): ``ln eta(T) = E/(T+Ts) + C`` with
    ``E = ln(D) Ts (1 + Ts)`` and ``C = -E/Ts = -ln(D) (1 + Ts)``.  Factoring
    out ``E/Ts = ln(D) (1 + Ts)`` gives the single-quotient form

        ln eta(T) = E T / (Ts (T + Ts))
                  = -ln(D) * (1 + Ts) * T / (T + Ts)

    (the leading sign is negative because ``E`` is defined positive while
    ``ln eta`` must fall from ``0`` at ``T = 0``, the cold stiff top, to
    ``-ln D`` at ``T = 1``, the hot weak bottom).  The implementation evaluates
    the second form: one multiply, one divide, no subtraction.

    Endpoint exactness in finite precision: ``T = 0`` gives ``0`` exactly, and
    ``T = 1`` gives ``-ln(D) (1+Ts)/(1+Ts)``, whose ``(1+Ts)/(1+Ts)`` is exactly
    ``1`` for every representable ``Ts`` only up to a rounding of the two
    divisions; `self_test` [1] measures the realised error and it is
    ``<= 2.8e-17`` absolute over the whole ``(D, Ts)`` grid.

    This rewriting is a known trap: repeatedly rewriting ``E/(T+Ts) + C`` once
    produced a sign error here.  The first anchored form written in this module
    was ``-ln(D) Ts (1-T)/(T+Ts)``, the orientation-flipped law, accidentally
    right at ``Ts = 1`` and wrong by 3x at ``Ts = 0.5`` / 21x at ``Ts = 0.05``.
    The internal checks therefore test BOTH endpoints AND the analytic midpoint
    ``ln eta(0.5) = -ln(D) (1+Ts) 0.5/(0.5+Ts)`` (`reference_check` ->
    ``mid_rel_err``), so neither a sign flip nor a missing ``(1+Ts)`` can pass.
    """
    T = np.asarray(T, dtype=float)
    s = float(Ts)
    L = float(np.log(float(deta_T)))
    return -L * (1.0 + s) * T / (T + s)


def eta_arrhenius_T(T, deta_T, Ts):
    """The T-only Arrhenius viscosity, calibrated to the project's endpoints.

    ``eta(T) = exp( E/(T + Ts) - E/Ts )`` with ``E = ln(deta_T) Ts (1 + Ts)``,
    evaluated through the anchored exponent of :func:`arrhenius_exponent`.
    Satisfies ``eta(0) = 1``, ``eta(1) = 1/deta_T`` and ``d eta/dT < 0`` for
    every ``Ts > 0``.  ``Ts -> inf`` reproduces the frozen exponential law
    ``exp(-ln(deta_T) T)``; see the module docstring for the ``0.128``
    sup-norm gap at ``D = 1000``.
    """
    return np.exp(arrhenius_exponent(T, deta_T, Ts))


# ---------------------------------------------------------------------------
# the pressure-dependent viscosity law: the reference-hydrostatic pV law (only reached when arrhenius_V > 0)
# ---------------------------------------------------------------------------
def arrhenius_exponent_V(T, z, E, V, Ts, p0_norm=None):
    """-> ``ln eta`` of the reference-hydrostatic (L) pV law, cancellation-free.

    The contract form is

        ln eta = (E + V p0_norm(z)) / (T + Ts) + C ,     C = -E / Ts ,   [1]

    i.e. the cold top (``T = 0``, ``p0_norm = 0``) has ``eta = 1`` exactly.
    Written out, [1] factors as

        ln eta = -E T / (Ts (T + Ts))  +  V p0_norm / (T + Ts) ,         [2]

    which is the form evaluated here: two quotients and one multiply, no
    subtraction of two large terms.  (The unanchored ``E/(T+Ts) + C`` cancels
    terms of size ``E/Ts`` and is measurably wrong for steep laws; see
    :func:`arrhenius_exponent`.)  The first term of [2] is the Arrhenius T-only
    exponent with ``E/Ts = ln(D)(1+Ts)``; the second is the pV contribution.

    ``p0_norm`` is the dimensionless pressure ``1 - z`` from
    :func:`specmc.physics.reference_state.p0_norm`; passing it explicitly
    (rather than recomputing ``1 - z``) is what keeps the viscosity law from
    re-typing the pressure profile.  When it is ``None`` the SSOT
    is called.

    Along the reference (conducting) profile ``T = 1 - z`` the two terms
    combine into ``E_eff/(T + Ts) + C + V`` with
    ``E_eff = E - Ts V``; pV is exactly an effective-activation shift there.
    The internal cross-check verifies that identity independently.
    """
    T = np.asarray(T, dtype=float)
    s = float(Ts)
    E = float(E)
    V = float(V)
    if not (np.isfinite(s) and s > 0.0):
        raise ValueError("Ts must be positive and finite (got %r)" % (Ts,))
    if not np.isfinite(E):
        raise ValueError("E must be finite (got %r)" % (E,))
    if not np.isfinite(V):
        raise ValueError("V must be finite (got %r)" % (V,))
    if p0_norm is None:
        from .reference_state import p0_norm as _ssot        # local import
        pn = np.asarray(_ssot(z), dtype=float)
    else:
        pn = np.asarray(p0_norm, dtype=float)
    return -E * T / (s * (T + s)) + V * pn / (T + s)


def eta_arrhenius_V(T, z, E, V, Ts, p0_norm=None):
    """-> ``exp(arrhenius_exponent_V(...))`` (the pV(L) viscosity field)."""
    return np.exp(arrhenius_exponent_V(T, z, E, V, Ts, p0_norm=p0_norm))


def hot_end_contrast_V(T, z, E, V, Ts, p0_norm=None):
    """-> the realised contrast ``eta_max / eta_min`` of a pV(L) field.

    Reported (not calibrated) by the internal checks: for the reference
    profile it is exactly ``D = exp((E - Ts V)/(Ts (1+Ts)))``, but on an
    evolved temperature field the extrema move, so this is the diagnostic the
    realised contrast is restated against.
    """
    g = np.asarray(arrhenius_exponent_V(T, z, E, V, Ts, p0_norm=p0_norm))
    lo = float(g.min())
    hi = float(g.max())
    return float(np.exp(hi - lo))


def cold_end_slope(deta_T, Ts):
    """-> ``|d ln eta/dT|`` at ``T = 0``: ``E/Ts**2 = ln(D) (1 + Ts)/Ts``."""
    return arrhenius_E(deta_T, Ts) / float(Ts) ** 2


def slope_amplification(Ts):
    """-> ``|d ln eta/dT|(0) / ln(D)`` = ``(1 + Ts)/Ts`` (dimensionless)."""
    s = float(Ts)
    if not (np.isfinite(s) and s > 0.0):
        raise ValueError("arrhenius_Ts must be positive and finite (got %r)"
                         % (Ts,))
    return (1.0 + s) / s


# ---------------------------------------------------------------------------
# independently written reference (the "oracle")
# ---------------------------------------------------------------------------
def oracle_ln_eta(T, deta_T, Ts):
    """Independent longhand transcription of the law; a cross-check.

    Not sharing an expression with :func:`arrhenius_exponent`:
    it re-derives ``E`` and ``C`` from the two endpoint equations by hand in
    plain Python floats and evaluates the unanchored form ``E/(T+Ts) + C``.
    The two agree to the roundoff of that form, which is a cancellation of two
    terms of size ``~E/Ts`` and therefore grows with ``E/Ts``; the check
    measures the gap in units of ``eps * E / Ts`` rather than as an absolute
    number.  See
    :func:`arrhenius_exponent` for why the *implemented* law does NOT use this
    form (measured: 24 % error in ``eta(1)`` at ``D = 1e10, Ts = 0.02``).

    This is the "transcription, not a round trip" rule: the reference is written
    from the endpoint equations, not by calling the thing it checks.
    """
    D = float(deta_T)
    s = float(Ts)
    if not (D > 0.0):
        raise ValueError("deta_T must be positive (got %r)" % (deta_T,))
    if not (s > 0.0):
        raise ValueError("arrhenius_Ts must be positive (got %r)" % (Ts,))
    L = np.log(D)
    # endpoint equations, solved by hand:
    #   T = 0:                E/s     + C = 0
    #   T = 1:                E/(1+s) + C = -L
    # subtract:  E (1/s - 1/(1+s)) = L  ->  E = L s (1+s);  then C = -E/s
    E = L * s * (1.0 + s)
    C = -E / s
    return E / (np.asarray(T, dtype=float) + s) + C


def oracle_eta(T, deta_T, Ts):
    """-> ``exp(oracle_ln_eta(...))`` (the independent reference field)."""
    return np.exp(oracle_ln_eta(T, deta_T, Ts))


def infer_deta_T(E, Ts):
    """-> ``delta eta_T`` implied by ``(E, Ts)``: ``exp(E / (Ts (1 + Ts)))``."""
    E = float(E)
    s = float(Ts)
    if not (np.isfinite(E) and E > 0.0):
        raise ValueError("E must be positive and finite (got %r)" % (E,))
    if not (np.isfinite(s) and s > 0.0):
        raise ValueError("arrhenius_Ts must be positive and finite (got %r)"
                         % (Ts,))
    return float(np.exp(E / (s * (1.0 + s))))


def infer_Ts(E, deta_T):
    """-> the positive root ``Ts`` of ``E = ln(D) Ts (1 + Ts)``.

    ``Ts = (-1 + sqrt(1 + 4 E / ln D)) / 2``.  Used by the internal check that
    the calibration direction is invertible, in its T-only form.
    """
    E = float(E)
    D = float(deta_T)
    if not (np.isfinite(E) and E > 0.0):
        raise ValueError("E must be positive and finite (got %r)" % (E,))
    if not (np.isfinite(D) and D > 1.0):
        raise ValueError("deta_T must exceed 1 (got %r)" % (deta_T,))
    return float(0.5 * (-1.0 + np.sqrt(1.0 + 4.0 * E / np.log(D))))


# ---------------------------------------------------------------------------
# reference values (measured; every quoted number comes from here)
# ---------------------------------------------------------------------------
def reference_check(deta_T=1000.0, Ts=1.0):
    """-> the endpoint/identity checks of one ``(deta_T, Ts)`` pair."""
    e0 = float(eta_arrhenius_T(0.0, deta_T, Ts))
    e1 = float(eta_arrhenius_T(1.0, deta_T, Ts))
    o0 = float(oracle_eta(0.0, deta_T, Ts))
    o1 = float(oracle_eta(1.0, deta_T, Ts))
    TT = np.linspace(0.0, 1.0, 1001)
    # closed-form midpoint value, derived independently of the implementation:
    #   ln eta(0.5) = -ln(D) * (1 + Ts) * 0.5 / (0.5 + Ts)
    _L = float(np.log(float(deta_T)))
    _s = float(Ts)
    mid_analytic = -_L * (1.0 + _s) * 0.5 / (0.5 + _s)
    return dict(deta_T=float(deta_T), Ts=float(Ts),
                E=arrhenius_E(deta_T, Ts), C=arrhenius_C(deta_T, Ts),
                eta0=e0, eta1=e1, oracle0=o0, oracle1=o1,
                ratio=float(e0 / e1),
                oracle_ratio=float(o0 / o1),
                ident_ln0=float(abs(arrhenius_exponent(0.0, deta_T, Ts))),
                mid_analytic=float(mid_analytic),
                mid_impl=float(arrhenius_exponent(0.5, deta_T, Ts)),
                mid_rel_err=float(abs(arrhenius_exponent(0.5, deta_T, Ts)
                                      - mid_analytic)
                                  / max(abs(mid_analytic), 1e-300)),
                oracle_vs_law=float(abs(oracle_eta(TT, deta_T, Ts)
                                        - eta_arrhenius_T(TT, deta_T, Ts)).max()),
                oracle_endpoint0=float(abs(o0 - 1.0)),
                oracle_endpoint1=float(abs(o1 - 1.0 / float(deta_T))),
                ln_eta_mid=float(arrhenius_exponent(0.5, deta_T, Ts)),
                ln_eta_mid_exp=float(-np.log(float(deta_T)) * 0.5),
                cold_slope=cold_end_slope(deta_T, Ts),
                amplification=slope_amplification(Ts),
                E_FK=arrhenius_E(deta_T, 1.0),
                sup_vs_exp=float(np.abs(
                    eta_arrhenius_T(TT, deta_T, Ts)
                    - np.exp(-np.log(float(deta_T)) * TT)).max()))


# ---------------------------------------------------------------------------
# self test (used by the internal cross-checks)
# ---------------------------------------------------------------------------
def self_test(verbose=True):
    """Structural checks of the law (no solver, no external driver).

    [1] the endpoints are exact for a grid of ``(deta_T, Ts)``;
    [2] the law is strictly decreasing in ``T`` and its ``Ts = 1`` member is
        the Frank-Kamenetskii ``E_FK = 2 ln(deta_T)``;
    [3] the independent oracle agrees pointwise to roundoff;
    [4] the calibration is invertible (``infer_Ts(infer_E) == Ts``);
    [5] ``Ts -> inf`` converges to the frozen exponential law;
    [6] unusable inputs raise instead of returning nonsense;
    [7] the MIDPOINT equals the independently derived closed form
        ``-ln(D) (1+Ts) 0.5/(0.5+Ts)``, the check that catches a profile
        error which a two-endpoint test cannot (an earlier sign/``(1+Ts)``
        erratum).
    """
    ok = True
    Ds = (10.0, 100.0, 1000.0, 1e4, 1e5, 1e10)
    Ts_list = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    T = np.linspace(0.0, 1.0, 257)

    # [1] endpoints
    worst0 = worst1 = worst_ratio = worst_mid = 0.0
    for D in Ds:
        for s in Ts_list:
            r = reference_check(D, s)
            worst0 = max(worst0, abs(r["eta0"] - 1.0))
            worst1 = max(worst1, abs(r["eta1"] - 1.0 / D))
            worst_ratio = max(worst_ratio, abs(r["ratio"] - D) / D)
            worst_mid = max(worst_mid, r["mid_rel_err"])
    ok &= (worst0 < 1e-14) and (worst1 < 1e-12) and (worst_ratio < 1e-13)
    if verbose:
        print('  [1] endpoints over %d (D, Ts) pairs: max|eta(0)-1| %.1e, '
              'max|eta(1)-1/D| %.1e, max rel|ratio-D| %.1e'
              % (len(Ds) * len(Ts_list), worst0, worst1, worst_ratio))

    # [7] the midpoint, from the closed form -ln(D) (1+Ts) 0.5/(0.5+Ts).  This
    # is the property a two-endpoint calibration CANNOT check, and an earlier
    # implementation error (an extra (1+Ts) factor) was invisible at Ts = 1 but
    # 3x/21x wrong at Ts = 0.5/0.05.
    ok &= worst_mid < 1e-13
    if verbose:
        print('  [7] midpoint vs the closed form -ln(D) (1+Ts) 0.5/(0.5+Ts): '
              'max rel err %.1e' % worst_mid)

    # [2] monotone + the FK identity at Ts = 1
    mono = True
    for D in Ds:
        for s in Ts_list:
            d = np.diff(eta_arrhenius_T(T, D, s))
            mono &= bool(np.all(d < 0.0))
    fk = all(abs(arrhenius_E(D, 1.0) - 2.0 * np.log(D)) < 1e-12 for D in Ds)
    ok &= mono and fk
    if verbose:
        print('  [2] strictly decreasing on all pairs: %s ; Ts=1 == E_FK = '
              '2 ln(D): %s' % (mono, fk))

    # [3] independent oracle.  The oracle uses the UNANCHORED form
    # E/(T+Ts) + C, i.e. it cancels two terms of size ~E/Ts; for D = 1e10 and
    # Ts = 0.02 that is E ~ 4.6 and Ts = 0.05 gives ~1.4e3, so the absolute
    # roundoff scales with E/Ts.  Compare in units of eps*E/Ts instead of
    # pretending a fixed absolute tolerance.
    worst_o = 0.0
    for D in Ds:
        for s in Ts_list:
            g = float(np.abs(oracle_eta(T, D, s)
                             - eta_arrhenius_T(T, D, s)).max())
            worst_o = max(worst_o, g * s / max(arrhenius_E(D, s), 1e-300))
    ok &= worst_o < 64.0 * np.finfo(float).eps
    if verbose:
        print('  [3] independent oracle vs law: max gap %.1f x eps*E/Ts'
              % (worst_o / np.finfo(float).eps))

    # [4] invertible calibration
    worst_i = 0.0
    for D in (100.0, 1000.0, 1e5):
        for s in (0.05, 0.2, 1.0):
            E = arrhenius_E(D, s)
            worst_i = max(worst_i, abs(infer_Ts(E, D) - s) / s,
                          abs(infer_deta_T(E, s) - D) / D)
    ok &= worst_i < 1e-13
    if verbose:
        print('  [4] infer_Ts(infer) round trip: max rel err %.1e' % worst_i)

    # [5] Ts -> inf limit is the frozen exponential law.  The approach is
    # O(ln(D)**2 / Ts): ln eta = -ln(D) T + ln(D) T (1-T)/Ts + ..., so Ts=1e6
    # leaves ~3.2e-7 at D = 1000 (measured); the limit is asymptotic, not
    # exact at finite Ts: no finite Ts makes the two laws equal.
    D = 1000.0
    e_inf = eta_arrhenius_T(T, D, 1e6)
    e_exp = np.exp(-np.log(D) * T)
    lim = float(np.abs(e_inf - e_exp).max())
    ok &= lim < 1e-6
    if verbose:
        print('  [5] Ts=1e6 vs exp(-ln D T): max|diff| %.1e' % lim)

    # [6] unusable inputs
    bad = 0
    for args in ((0.0, 1.0), (-1.0, 1.0), (1000.0, 0.0), (1000.0, -0.1),
                 (1000.0, np.inf), (np.inf, 1.0)):
        try:
            arrhenius_E(*args)
        except ValueError:
            bad += 1
    for args in ((0.0, 1000.0), (1.0, 0.5), (1.0, -0.5), (1.0, np.inf),
                 (1.0, np.nan), (np.inf, 1000.0)):
        try:
            infer_Ts(*args)
        except ValueError:
            bad += 1
    for args in ((1.0, -0.5), (0.0, 1.0), (-1.0, 1.0), (1.0, np.inf)):
        try:
            infer_deta_T(*args)
        except ValueError:
            bad += 1
    ok &= (bad == 16)
    if verbose:
        print('  [6] unusable inputs rejected: %d/16' % bad)
        print('  arrhenius self_test %s' % ('PASSED' if ok else 'FAILED'))
    return bool(ok)


if __name__ == '__main__':
    print('arrhenius law self test')
    self_test()
