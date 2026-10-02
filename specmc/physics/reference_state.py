"""Reference state: the single source of truth for the pV(L) law.

What this module is
-------------------
The project is dimensionless: the frozen reference interface exposes ``Ra``, ``H`` and
``deta_T`` and no length, density or gravity scale at all.  The ``pV`` term
of the full Arrhenius exponent,

    eta = eta_0 exp( (E_a + p V_a) / (R T) ) ,                    [1]

is the first physics this project adds that *needs* those scales, and the change
that adds it must therefore also define, once and in one place, which
pressure it means and which dimensional constants anchor the dimensionless
groups.  That is what this module owns.  Nothing else in the package may
re-type ``rho g z``: the viscosity law consumes
:func:`p0_norm`, not a hand-written ``1 - z``.

The reference hydrostatic pressure
----------------------------------
The frozen pV contract (`specmc.physics.arrhenius.PV_CONTRACT`) fixes
the pressure to be the reference (hydrostatic) pressure

    p0(z) = rho0 g d (1 - z) ,      p0_norm(z) = p0 / (rho0 g d) = 1 - z ,  [2]

with ``z = 1`` the cold top and ``z = 0`` the hot bottom.  ``pV = 0`` at the
top and maximal at the bottom.  This is the reference-pressure contract of
the pV law; the two mature implementations of it are

* CitcomS ``rheol=4``, ``visc = visc0 exp((viscE + viscZ(1-r))/(T*+viscT))``
  with ``P = rho g (1-r)`` the *hydrostatic* pressure and
  ``viscZ = rho g V_a/(R DeltaT)`` (`_lit/citcoms-manual.tex:2683-2707`,
  `Viscosity_structures.c:709-740`);
* ASPECT ``frank_kamenetskii.cc:56-57``, ``pressure_prefactor *
  (p - p_ref)/(density*gravity*max_depth)``: the same normalisation,
  ``(p - p_ref)/(rho g d) = 1 - z`` for a surface-referenced hydrostatic
  profile.

The two remaining pressure conventions are explicitly OUT of this module
and out of ``arrhenius_V``'s meaning: the dynamic pressure ``p_d`` the Stokes solve
produces and the total pressure ``p0 + p_d`` (``specmc/config.py``
"Frozen pV contract").  Both need an outer eta-p fixed point and are a
different change.

The dimensional constants contract
----------------------------------
Eight constants, each with a provenance, frozen in :class:`ReferenceConstants`:

    rho0    reference density             3300 kg m^-3   ASPECT king_plugin.cc:163
    g       gravity                       9.81 m s^-2    standard gravity
    d       layer depth                   2.89e6 m       ASPECT manual :24665-24669
    DeltaT  temperature drop              3000 K         king_ala.prm:10-13
    T_surf  surface temperature           273 K          king_ala.prm:10-13
    V_a     activation volume (diffusion) 6e-6 m^3/mol   ASPECT manual :24646-24651
    E_a     activation energy (Peierls)   5.3e5 J/mol    ASPECT manual :24630-24631
    R       gas constant                  8.314462618 J/(mol K)   CODATA 2018

They define exactly two dimensionless groups and one activation:

    V-tilde = rho0 g d V_a / (R DeltaT)      (pV strength)
    Ts      = T_surf / DeltaT                (surface temperature offset)
    E-tilde = E_a / (R DeltaT)               (activation)

``V-tilde/E-tilde = rho0 g d V_a / E_a = 1.059`` for these defaults, exactly
independent of DeltaT, the physical, dimensionless number that none of
``{Ra, H, deta_T}`` contains.

The calibration, and one arithmetic correction
-----------------------------------------------
Along the *reference (conducting) profile* ``T = 1 - z`` the law [1] collapses:

    ln eta(T, z) = (E-tilde + V-tilde (1-z)) / (T + Ts) + C
                 = [ E-tilde - Ts V-tilde ] / (T + Ts) + (C + V-tilde)
                 = E_eff / (T + Ts) + C' ,      E_eff := E-tilde - Ts V-tilde

i.e. pV is exactly an effective-activation shift along the reference
profile.  This identity is the algebraic core of the calibration and of
the independent longhand reference.

The counting matters, and a first pass at this calibration states it
incorrectly.  There are four unknowns
``(E-tilde, V-tilde, Ts, C)`` and only two equations (the cold-end
normalisation ``eta = 1`` at ``T = 0`` fixes ``C = -E-tilde/Ts``; the hot-end
value *defines* ``ln D``).  The two equations therefore determine ``C`` and the
combination ``E_eff``, not ``E-tilde``.  Anchoring ``V-tilde`` and ``Ts``
on the physical constants still leaves one more anchor:

    * anchor ``"Ea"``  : E-tilde = E_a/(R DeltaT)  ->  ``D`` is the OUTPUT;
    * anchor ``"D"``   : ``D = deta_T`` given       ->  E-tilde (and the
                          implied ``E_a``) is the output.

Exactly one of the two may be used (:func:`calibrate_pv` enforces that).  The
first-pass table quotes ``E-tilde ~ 145.8``, ``V-tilde ~ 154.5`` and
``D ~ 1.1e4``; all three are arithmetically wrong:

    * the physical ``V-tilde`` is 22.505, not 154.5 (``rho0 g d V_a/(R DeltaT)``);
    * ``E-tilde = E_a/(R DeltaT) = 21.248``, not 145.8;
    * with the physical anchors ``D = exp(E_eff/(Ts (1+Ts))) = e^{193.39} =
      9.75e83``, not ``1.1e4``; that triple
      ``(145.8, 154.5, 0.091)`` would give ``e^{1326.9}``.

The large contrast is *physics*, not a bug: a raw Arrhenius law over
``T = 273 .. 3273 K`` with ``E_a = 5.3e5 J/mol`` does span ``e^{214}``.
That is exactly why mature codes calibrate a Frank-Kamenetskii
(linearised) activation instead of using the raw one.  The anchor ``"D"``
(``E-tilde = ln(deta_T) Ts (1+Ts) + Ts V-tilde``, so the reference contrast is kept
and pV redistributes it with depth) is implemented as the first-class
alternative; ``E_a`` is then an *output*.

TALA (reserved, NOT wired yet)
-------------------------------------
``rho_bar`` / ``T_bar`` / ``p_bar_king`` reproduce the King et al. (2010)
reference profiles *algebraically* so that a later TALA implementation has its single
source of truth already in place.  This check only compares them against the
published formulas (the equation layer is a separate change:
``grad.(rho_bar u) = 0`` would break
the per-wavenumber elimination of `specmc.stokes.stokes_coupled`).

DOI
---
King, Lee, Van Keken, Leng, Zhong, Tan, Tosi & Kameyama (2010), GJI 180(1),
73-87, ``10.1111/j.1365-246X.2009.04403.x``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

__all__ = [
    "ReferenceConstants", "REFERENCE_DEFAULTS",
    "p0_norm", "p0", "V_tilde", "Ts_tilde", "E_tilde",
    "infer_V_a", "infer_E_a", "calibrate_pv", "PV_ANCHORS",
    "rho_bar", "T_bar", "p_bar_king",
    "KING_ALA_DI1_RA1E4", "self_test",
]


# ---------------------------------------------------------------------------
# the frozen dimensional-constants contract
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReferenceConstants:
    """The eight dimensional constants, each with a provenance (docstring).

    Frozen on purpose: a dataclass instance is the *contract*, and
    :func:`calibrate_pv` refuses anything that is not finite and positive.  The
    defaults are ASPECT's published defaults for an Earth-like mantle
    (``king_plugin.cc:163``, the ASPECT manual ``:24630-24669``,
    ``king_ala.prm:10-13``); they are NOT tuned to this project.

    ``E_a`` is the Peierls-creep default ``5.3e5 J/mol``; the diffusion-creep
    activation volume ``6e-6 m^3/mol`` is the matching ``V_a`` (the pair used
    for ``pV_a/E_a = 1.06``).
    """
    #: reference (mantle) density, kg m^-3, ASPECT king_plugin.cc:163
    rho0: float = 3300.0
    #: gravitational acceleration, m s^-2, standard gravity
    g: float = 9.81
    #: layer depth, m, ASPECT manual :24665-24669 (32436 Pa/m gradient)
    d: float = 2.89e6
    #: temperature drop across the layer, K, king_ala.prm:10-13
    DeltaT: float = 3000.0
    #: surface temperature, K, king_ala.prm:10-13 (0.091 * 3000 = 273)
    T_surf: float = 273.0
    #: activation volume (diffusion creep), m^3 mol^-1, manual :24646-24651
    V_a: float = 6e-6
    #: activation energy (Peierls creep), J mol^-1, manual :24630-24631
    E_a: float = 5.3e5
    #: molar gas constant, J mol^-1 K^-1, CODATA 2018
    R: float = 8.314462618

    def as_dict(self):
        return asdict(self)


#: the single frozen instance every call site uses unless it passes its own
REFERENCE_DEFAULTS = ReferenceConstants()

#: the two legal calibration anchors (exactly one is used per calibration)
PV_ANCHORS = ("Ea", "D")


def _finite_positive(name, value):
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError("ReferenceConstants.%s must be positive and finite "
                         "(got %r)" % (name, value))
    return v


def _check_constants(c):
    if not isinstance(c, ReferenceConstants):
        raise TypeError("expected ReferenceConstants, got %r" % (type(c),))
    for name in ("rho0", "g", "d", "DeltaT", "T_surf", "E_a", "R"):
        _finite_positive(name, getattr(c, name))
    # V_a = 0 is legal and means "pV off" (it is the internal consistency check
    # of the whole calibration: the V = 0 limit must reproduce the Arrhenius law exactly).
    v_a = float(c.V_a)
    if not np.isfinite(v_a) or v_a < 0.0:
        raise ValueError("ReferenceConstants.V_a must be finite and >= 0 "
                         "(got %r)" % (c.V_a,))
    return c


# ---------------------------------------------------------------------------
# the reference hydrostatic pressure (the pV single source of truth)
# ---------------------------------------------------------------------------
def p0_norm(z):
    """-> the dimensionless reference hydrostatic pressure ``1 - z``.

    This is the ONLY definition of the pressure profile in the package:
    ``z = 1`` is the cold top (``p0 = 0``), ``z = 0`` the hot bottom
    (``p0 = rho0 g d``).  Exact in floating point at both ends and strictly
    decreasing in ``z``.
    """
    return 1.0 - np.asarray(z, dtype=float)


def p0(z, constants=None):
    """-> the dimensional reference hydrostatic pressure ``rho0 g d (1 - z)``.

    The pV law itself never calls this: it consumes :func:`p0_norm`, so that
    ``rho0 g d`` appears exactly once (in ``V-tilde``) and no viscosity module
    re-types the product.
    """
    c = _check_constants(constants or REFERENCE_DEFAULTS)
    return c.rho0 * c.g * c.d * p0_norm(z)


def V_tilde(constants=None):
    """-> ``V-tilde = rho0 g d V_a / (R DeltaT)`` (the pV strength).

    ``22.504874770248204`` for :data:`REFERENCE_DEFAULTS`.  Its ratio to
    :func:`E_tilde` is ``rho0 g d V_a / E_a`` and is independent of
    ``DeltaT``.
    """
    c = _check_constants(constants or REFERENCE_DEFAULTS)
    return c.rho0 * c.g * c.d * c.V_a / (c.R * c.DeltaT)


def Ts_tilde(constants=None):
    """-> ``Ts = T_surf / DeltaT`` (the temperature offset in ``T + Ts``).

    ``0.091`` for :data:`REFERENCE_DEFAULTS`, exactly King's nondimensional
    surface temperature ``king_ala.prm:13``.
    """
    c = _check_constants(constants or REFERENCE_DEFAULTS)
    return c.T_surf / c.DeltaT


def E_tilde(constants=None):
    """-> ``E-tilde = E_a / (R DeltaT)`` (the dimensionless activation)."""
    c = _check_constants(constants or REFERENCE_DEFAULTS)
    return c.E_a / (c.R * c.DeltaT)


def infer_V_a(V, constants=None):
    """-> the activation volume implied by a given ``V-tilde``.

    ``V_a = V R DeltaT / (rho0 g d)``.  Used by an internal cross-check to
    prove that a physical ``(E_a, V_a, rho0, g, d, DeltaT)`` set can be
    inverted to the same ``V-tilde``.
    """
    c = _check_constants(constants or REFERENCE_DEFAULTS)
    return float(V) * c.R * c.DeltaT / (c.rho0 * c.g * c.d)


def infer_E_a(E, constants=None):
    """-> the activation energy implied by a given ``E-tilde``."""
    c = _check_constants(constants or REFERENCE_DEFAULTS)
    return float(E) * c.R * c.DeltaT


# ---------------------------------------------------------------------------
# the calibration, the single procedure every pV call site shares
# ---------------------------------------------------------------------------
def calibrate_pv(constants=None, anchor="Ea", deta_T=None, V=None):
    """-> the frozen pV calibration record for one anchor choice.

    The law (:data:`specmc.physics.arrhenius.PV_CONTRACT`) is

        ln eta(T, z) = (E-tilde + V-tilde p0_norm(z)) / (T + Ts) + C ,
        C = -E-tilde / Ts                                    (cold top: eta = 1)

    and along the reference profile ``T = 1 - z`` it reduces to
    ``E_eff/(T + Ts) + C'`` with ``E_eff = E-tilde - Ts V-tilde`` and
    ``C' = C + V-tilde`` (the module docstring derives this).

    ``anchor`` selects the third, independent condition (exactly one):

    ``"Ea"``
        ``E-tilde = E_a/(R DeltaT)`` from the constants contract, so the
        contrast is an output: ``D = exp(E_eff/(Ts (1+Ts)))``.  Requires no
        ``deta_T``.
    ``"D"``
        ``D = deta_T`` (the frozen reference contrast) is the input, so
        ``E_eff = ln(deta_T) Ts (1+Ts)`` and
        ``E-tilde = E_eff + Ts V-tilde``; the implied ``E_a`` is an output.
        This keeps ``deta_T`` meaningful under pV.

    ``V`` overrides the physical ``V-tilde = rho0 g d V_a/(R DeltaT)`` with an
    explicit (non-negative, finite) coefficient, that is how a ``V``-scan
    factor is expressed without touching the constants contract.

    Both anchors are computed and cross-checked against each other: the
    ``"Ea"`` record contains the ``D`` it implies and the ``"D"`` record the
    ``E_a`` it implies, so the calling code can report either without a second
    call.

    Returns a plain dict (JSON-safe) with ``E``, ``V``, ``Ts``, ``C``,
    ``E_eff``, ``D``, ``ln_D``, ``Ea``, ``V_a``, ``anchor`` and the constants.
    Raises ``ValueError`` for an unknown anchor, a missing ``deta_T`` under the
    ``"D"`` anchor, a ``deta_T <= 1`` (a contrast of 1 or less is not a
    calibrated law), or a non-finite/negative ``V``.
    """
    c = _check_constants(constants or REFERENCE_DEFAULTS)
    a = str(anchor)
    if a not in PV_ANCHORS:
        raise ValueError("anchor must be one of %r (got %r)"
                         % (PV_ANCHORS, anchor))
    V = V_tilde(c) if V is None else float(V)
    if not np.isfinite(V) or V < 0.0:
        raise ValueError("V-tilde must be finite and >= 0 (got %r)" % (V,))
    Ts = Ts_tilde(c)
    if a == "Ea":
        E = E_tilde(c)
        E_eff = E - Ts * V
        # cold-top normalisation put eta = 1 at T = 0 (p0_norm = 0); the hot
        # bottom (T = 1, p0_norm = 1) therefore has
        #     ln eta(1) = (E + V)/(1+Ts) - E/Ts = -E_eff/(Ts(1+Ts)) = -ln D
        ln_D = E_eff / (Ts * (1.0 + Ts))
        D = float(np.exp(ln_D))
        Ea = c.E_a
    else:
        if deta_T is None:
            raise ValueError('anchor="D" needs deta_T (the frozen contrast)')
        D = float(deta_T)
        if not (np.isfinite(D) and D > 1.0):
            raise ValueError("deta_T must exceed 1 under anchor='D' (got %r)"
                             % (deta_T,))
        ln_D = float(np.log(D))
        E_eff = ln_D * Ts * (1.0 + Ts)
        E = E_eff + Ts * V
        Ea = infer_E_a(E, c)
    C = -E / Ts
    rec = dict(anchor=a, law="pV(L) reference hydrostatic",
               E=float(E), V=float(V), Ts=float(Ts), C=float(C),
               E_eff=float(E_eff), ln_D=float(ln_D), D=float(D),
               Ea=float(Ea), V_a=float(c.V_a), R=float(c.R),
               rho0=float(c.rho0), g=float(c.g), d=float(c.d),
               DeltaT=float(c.DeltaT), T_surf=float(c.T_surf),
               E_a=float(c.E_a), ratio_VE=float(V / E),
               p0_ref=float(c.rho0 * c.g * c.d),
               contrast_log10=float(ln_D / np.log(10.0)))
    # the identity that the whole calibration rests on: the hot-bottom value
    # must be exactly -ln D (up to the roundoff of this expression)
    chk = (rec["E"] + rec["V"]) / (1.0 + rec["Ts"]) + rec["C"]
    rec["ln_eta_hot_end"] = float(chk)
    rec["hot_end_residual"] = float(abs(chk + rec["ln_D"]))
    return rec


# ---------------------------------------------------------------------------
# TALA interface (reserved, NOT wired to any equation yet)
# ---------------------------------------------------------------------------
def _depth(z):
    return 1.0 - np.asarray(z, dtype=float)


def rho_bar(z, Di, gamma=1.0):
    """-> King's reference density ``rho_bar = exp(depth Di / gamma)``.

    ``king_ala.prm:50`` (third expression), ``king_plugin.cc:80``.  ``depth``
    is ``1 - z`` (``king_plugin.cc:72``: ``depth = 1 - position(dim-1)``, with
    ASPECT's ``z`` increasing upward, the same convention as this project's).
    Rows with ``Di = 0`` give ``rho_bar = 1``.
    """
    Di = float(Di)
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be positive and finite (got %r)"
                         % (gamma,))
    return np.exp(_depth(z) * Di / gamma)


def T_bar(z, Di, T0=0.091):
    """-> King's reference temperature ``T_bar = T0 exp(depth Di)``.

    ``king_ala.prm:50`` (first expression).  ``T0 = 0.091`` is King's
    nondimensional surface temperature; ``Di = 0`` gives the constant profile.
    """
    return float(T0) * np.exp(_depth(z) * float(Di))


def p_bar_king(z, Di, gamma=1.0):
    """-> King's reference pressure ``p_bar = gamma/Di (exp(depth Di/gamma)-1)``.

    ``king_ala.prm:50`` (second expression).  Not the pV pressure: the
    reference hydrostatic pressure uses :func:`p0_norm` (``1 - z``), the
    Boussinesq/incompressible
    hydrostatic profile.  This one is the *anelastic* reference pressure and is
    provided only for the reserved TALA interface.  ``Di -> 0`` is the limit
    ``p_bar -> depth``.
    """
    Di = float(Di)
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be positive and finite (got %r)"
                         % (gamma,))
    if Di == 0.0:
        return _depth(z).copy()
    return gamma / Di * (np.exp(_depth(z) * Di / gamma) - 1.0)


#: King et al. (2010) ASPECT reference row, ALA ``Di = 1.0``, ``Ra = 1e4``
#: (`_extC_src/aspect/king_reference_statistics.txt:18`).  Recorded here so the
#: read-out can quote the published numbers without re-parsing the file.
KING_ALA_DI1_RA1E4 = dict(
    formulation="ALA", Di=1.0, Ra=1e4, code="ASPECT",
    Nu=2.4459907, Nu_g=2.4459761, Vrms=24.680937, T_mean=0.5114233,
    Phi=1.3426752, W=1.3539737,
    source="ASPECT benchmarks/king2dcompressible/reference_statistics.txt:18",
    doi="10.1111/j.1365-246X.2009.04403.x",
)


# ---------------------------------------------------------------------------
# self test (structural checks of the reference state)
# ---------------------------------------------------------------------------
def self_test(verbose=True):
    """Structural checks of the reference state and the calibration.

    [1] the reference hydrostatic pressure: ``p0_norm(1) = 0`` and
        ``p0_norm(0) = 1`` exactly, strictly decreasing, and
        ``p0(0) = rho0 g d`` bit-exactly;
    [2] the dimensionless groups have the documented values and
        ``V-tilde/E-tilde`` is independent of DeltaT;
    [3] the ``"D"`` anchor with ``V-tilde = 0`` reproduces the Arrhenius T-only
        activation ``ln(deta_T) Ts (1+Ts)`` exactly, so the pV machinery is a
        strict extension of the frozen law;
    [4] the effective-activation identity
        ``E_eff = E-tilde - Ts V-tilde`` and, independently, that the law's
        hot-end value is exactly ``-ln D``;
    [5] the two anchors are mutual inverses: ``Ea -> D -> (D anchor) -> Ea``;
    [6] the King profiles reproduce ``king_ala.prm:50`` at several depths, and
        the ``Di = 0`` limits are the documented constants;
    [7] unusable inputs raise instead of returning nonsense.
    """
    ok = True
    c = REFERENCE_DEFAULTS

    # [1] the reference hydrostatic pressure
    z = np.linspace(0.0, 1.0, 257)
    pn = p0_norm(z)
    e_top = float(p0_norm(1.0))
    e_bot = float(p0_norm(0.0))
    mono = bool(np.all(np.diff(pn) < 0.0))
    exact = float(p0(0.0, c)) == float(c.rho0 * c.g * c.d)
    ok &= (e_top == 0.0) and (e_bot == 1.0) and mono and exact
    if verbose:
        print("  [1] p0_norm(1)=%.1f p0_norm(0)=%.1f monotone=%s "
              "p0(0)==rho0 g d bit-exact=%s"
              % (e_top, e_bot, mono, exact))

    # [2] groups + the DeltaT independence of the ratio
    V, Ts, E = V_tilde(c), Ts_tilde(c), E_tilde(c)
    ratio = V / E
    ratio_ref = c.rho0 * c.g * c.d * c.V_a / c.E_a
    worst = 0.0
    for dT in (1000.0, 3000.0, 1e4):
        cc = replace(c, DeltaT=dT)
        worst = max(worst, abs((V_tilde(cc) / E_tilde(cc) - ratio_ref)
                               / ratio_ref))
    ok &= (abs(ratio - ratio_ref) < 1e-15) and (worst < 1e-14)
    if verbose:
        print("  [2] V-tilde=%.15g  Ts=%.15g  E-tilde=%.15g  "
              "V/E=%.15g (physical %.15g, DeltaT spread %.1e)"
              % (V, Ts, E, ratio, ratio_ref, worst))

    # [3] V = 0 + anchor D reproduces the Arrhenius law activation
    for D in (10.0, 1000.0, 1e4, 1e10):
        rec = calibrate_pv(c, anchor="D", deta_T=D)
        E_extB = float(np.log(D) * Ts * (1.0 + Ts))
        zero = replace(c, V_a=0.0)
        rec0 = calibrate_pv(zero, anchor="D", deta_T=D)
        ok &= abs(rec0["E"] - E_extB) < 1e-14 * abs(E_extB)
        ok &= abs(rec["E"] - (E_extB + Ts * V)) < 1e-14 * abs(E_extB + Ts * V)
    if verbose:
        print("  [3] anchor='D' with V=0 reproduces the Arrhenius E = ln(D) Ts (1+Ts) "
              "for D in {10, 1e3, 1e4, 1e10}")

    # [4] the effective-activation identity + the hot end
    from .arrhenius import arrhenius_exponent_V                  # local import
    TT = np.linspace(0.0, 1.0, 101)
    rec = calibrate_pv(c, anchor="Ea")
    zz = 1.0 - TT                       # the reference (conducting) profile
    lhs = np.asarray(arrhenius_exponent_V(TT, zz, rec["E"], rec["V"],
                                          rec["Ts"]), dtype=float)
    rhs = rec["E_eff"] / (TT + rec["Ts"]) + rec["C"] + rec["V"]
    gap = float(np.abs(lhs - rhs).max())
    hot = float(arrhenius_exponent_V(1.0, 0.0, rec["E"], rec["V"],
                                     rec["Ts"]))
    ok &= (gap < 1e-9) and (abs(hot + rec["ln_D"]) < 1e-10 * abs(rec["ln_D"]))
    if verbose:
        print("  [4] E_eff identity on the reference profile: max gap %.1e ; "
              "hot end ln eta = %.6f vs -ln D = %.6f"
              % (gap, hot, -rec["ln_D"]))

    # [5] the two anchors are mutual inverses
    recA = calibrate_pv(c, anchor="Ea")
    recD = calibrate_pv(c, anchor="D", deta_T=recA["D"])
    ok &= abs(recD["E"] - recA["E"]) <= 1e-13 * abs(recA["E"])
    ok &= abs(recD["D"] - recA["D"]) <= 1e-13 * abs(recA["D"])
    ok &= abs(infer_V_a(V_tilde(c), c) - c.V_a) <= 1e-15 * c.V_a
    if verbose:
        print("  [5] anchor round trip: Ea -> D -> Ea  max rel err %.1e ; "
              "V_a round trip rel err %.1e"
              % (abs(recD["E"] - recA["E"]) / abs(recA["E"]),
                 abs(infer_V_a(V_tilde(c), c) - c.V_a) / c.V_a))

    # [6] King profiles vs king_ala.prm:50
    Di = 1.0
    gamma = 1.0
    dep = np.array([0.0, 0.25, 0.5, 1.0])
    zb = 1.0 - dep
    rho_ref = np.exp(dep * Di / gamma)
    T_ref = 0.091 * np.exp(dep * Di)
    p_ref = gamma / Di * (np.exp(dep * Di / gamma) - 1.0)
    g_rho = float(np.abs(rho_bar(zb, Di, gamma) - rho_ref).max())
    g_T = float(np.abs(T_bar(zb, Di, T0=0.091) - T_ref).max())
    g_p = float(np.abs(p_bar_king(zb, Di, gamma) - p_ref).max())
    g0 = (abs(float(rho_bar(0.3, 0.0)) - 1.0)
          + abs(float(T_bar(0.3, 0.0, T0=0.091)) - 0.091)
          + abs(float(p_bar_king(0.3, 0.0)) - 0.7))
    ok &= (g_rho < 1e-15) and (g_T < 1e-15) and (g_p < 1e-14) and (g0 < 1e-15)
    if verbose:
        print("  [6] King profiles vs king_ala.prm:50: rho %.1e, T %.1e, "
              "p %.1e ; Di=0 limits %.1e" % (g_rho, g_T, g_p, g0))

    # [7] unusable inputs
    bad = 0
    try:
        calibrate_pv(c, anchor="nope")
    except ValueError:
        bad += 1
    try:
        calibrate_pv(c, anchor="D")
    except ValueError:
        bad += 1
    try:
        calibrate_pv(c, anchor="D", deta_T=1.0)
    except ValueError:
        bad += 1
    try:
        calibrate_pv(c, anchor="D", deta_T=0.5)
    except ValueError:
        bad += 1
    for field in ("rho0", "g", "d", "DeltaT", "T_surf", "V_a", "E_a", "R"):
        try:
            calibrate_pv(replace(c, **{field: -1.0}), anchor="Ea")
        except ValueError:
            bad += 1
    try:
        calibrate_pv(replace(c, **{"rho0": float("nan")}), anchor="Ea")
    except ValueError:
        bad += 1
    try:
        calibrate_pv(object(), anchor="Ea")
    except TypeError:
        bad += 1
    try:
        rho_bar(0.5, 1.0, gamma=0.0)
    except ValueError:
        bad += 1
    ok &= (bad == 15)
    if verbose:
        print("  [7] unusable inputs rejected: %d/15" % bad)
        print("  reference_state self_test %s" % ("PASSED" if ok else "FAILED"))
    return bool(ok)


if __name__ == "__main__":
    print("reference_state self test")
    self_test()
