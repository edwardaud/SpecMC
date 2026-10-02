"""Public configuration for specmc (Spectral Mantle Convection).

Design rule
-----------
The public interface exposes the physical scenario plus the numerical
parameters that the scenario itself forces.  Every other numerical choice is
a fixed implementation decision and lives in `specmc/_numerics.py`; it is not
reachable from here on purpose (CFL policy, time scheme, pre/post-processing,
dealiasing path, preconditioner selection, defect-correction route, ...).

Frozen default
--------------
``ConvectionConfig()`` reproduces the frozen reference configuration: 64x96,
Ra = 1e4, eta = exp(-ln(1000) * T(x,z,t)), no internal heating, no viscous
dissipation, Boussinesq, defect-corrected coupled Stokes with lag = 16 /
K = 3 / defect_tol = 1e-8 / max_K = 8.  The default configuration is checked
key by key against the production driver defaults and the recorded reference
values.

Internal heating
---------------
``internal_heating`` is implemented: a nonzero value adds the constant
volumetric source H of `specmc.physics.heating` to the temperature equation
(``d(theta)/dt + u.grad(theta) - u_z = lap(theta) + H``), in the form used by
ASPECT (``Radiogenic heating rate``), CitcomS (``Q0``), Underworld
(``fn_sourceTerm`` / ``adv_diff.f``) and G-ADOPT (``H=``).  The default
``0.0`` is the frozen reference value and is bit-for-bit the old code path, so the
immune region is exact.

Viscosity bounds: hard clip
---------------------------
``eta_min`` / ``eta_max`` are implemented: the pointwise viscosity is clipped
into that window by ``specmc.physics.eta_bounds.clip_eta``, *in the padded space
and before the truncation back to the resolved modes* (the anti-aliasing
rule), and every driver records the number of moved points as ``n_clip_eta``.
The default ``(1e-3, 1e3)`` is the law's own calibration window
(``Delta eta_T = 1000`` either side of the reference viscosity ``eta_ref = 1``):
the frozen case-2a law spans ``[1e-3, 1]``, which is strictly inside it, so the
frozen reference configuration is bit-for-bit unaffected.

Viscosity bounds: smooth clip
-----------------------------
``eta_smooth`` replaces that hard clip by a C^2 compact-support smoothstep
clamp in ``ln eta`` (`specmc.physics.eta_bounds.smooth_clip_eta`): exactly
bound-preserving, monotone, pointwise, and, at ``eta_smooth = 0.0``, the
default, *bit-for-bit* the hard clip, because the zero-width branch is
taken before any log/exp.  The model records the number of points inside a
transition band as ``n_smooth_eta`` next to the unchanged ``n_clip_eta``.

Stokes preconditioner selector
-----------------------------
``precond`` selects the Stokes solve: ``'legacy'`` (the frozen reference lagged
factorisation + adaptive Picard defect correction, bit for bit) or ``'bfbt'``
(the spectral BFBT preconditioned FGMRES route of ``specmc.stokes.bfbt``).  Both
solve the same coupled Schur system; only the Krylov path differs.

Viscosity bounds: reachable window
----------------------------------
No new field: the internally heated cases use the widest window the
frozen ``eta(T)`` law can reach,
``specmc.physics.eta_bounds.reachable_window(deta_T)`` = ``(1e-6, 1e6)`` for
``deta_T = 1000`` (the default).  Its lower edge is the smallest viscosity the
frozen temperature clip ``ETA_T_CLIP = (0, 2)`` permits, so the window is a
bit-for-bit no-op on every state the solver can produce, it removes the
*extra*, physics-changing ``eta_min = 1e-2`` bound used earlier by
construction rather than by a new hand-picked number, while the default
``(1e-3, 1e3)`` is untouched, so the frozen reference immune region stays exact.

pV contract
-----------
``arrhenius_V`` is the dimensionless coefficient ``V-tilde`` of the reference
(hydrostatic) pressure term of the full Arrhenius exponent,

    V-tilde = rho0 * g * d * V_a / (R * Delta T)
    exponent = (E + V-tilde * p0(z)) / (T + Ts) + C ,     p0(z) = rho0 g d (1-z)

with ``rho0`` the reference density, ``g`` gravity, ``d`` the layer depth,
``V_a`` the activation volume, ``R`` the gas constant and ``Delta T`` the
temperature drop across the layer.  The default ``0.0`` means "pressure term
off".  The pressure-dependent viscosity law implements it under the frozen semantics:

  * the pressure is the reference hydrostatic ``p0(z)``, never the dynamic
    pressure the Stokes solve produces, and never their sum;
  * ``p0`` comes from the reference-state single source of truth
    (:mod:`specmc.physics.reference_state`); no viscosity module re-types
    ``rho g z``;
  * ``arrhenius_V == 0.0`` is a bit-for-bit no-op of the T-only law: the
    model takes the pV branch only for a strictly positive value;
  * with ``V != 0`` the two endpoint conditions do not determine
    ``(E, V)``, so the calibration is the ONE procedure
    :func:`specmc.physics.reference_state.calibrate_pv`: ``V-tilde`` and
    ``Ts`` come from the eight-constant contract, and exactly one of
    ``E-tilde`` (physical, the model default) / the contrast ``D`` is the third
    anchor.  The default is the physical branch, so the contrast is an
    output, with the ASPECT defaults ``D = e^{193.39}``, not the ``1.1e4``
    quoted earlier (see the `reference_state` module docstring for the
    corrected arithmetic);
  * ``arrhenius_V < 0`` is rejected (it cannot be produced by the physical
    definition above).

The physically required magnitude is ``p V_a / E_a = 1.06`` (diffusion) to
``2.47`` (dislocation/Peierls) for ASPECT's default activation volumes, i.e.
``O(1)``.

Arrhenius law
-------------
``viscosity_model='arrhenius'`` implements the T-only activated-creep law of
`specmc.physics.arrhenius`::

    eta(T) = exp( E/(T + Ts) - E/Ts ),    E = ln(deta_T) * Ts * (1 + Ts)

with the temperature offset ``arrhenius_Ts``.  It satisfies the SAME two
endpoint conditions as the frozen exponential law (``eta(0) = 1`` at the cold
top, ``eta(1) = 1/deta_T`` at the hot bottom) for EVERY ``Ts > 0``, so the two
laws differ in the PROFILE, not in the endpoints; ``Ts -> inf`` recovers the
exponential law.  ``arrhenius_E`` is an optional echo of the calibrated
activation (a nonzero value must equal ``ln(deta_T) Ts (1 + Ts)``).  The
default ``arrhenius_Ts = 0.0`` means "Arrhenius off": the model then runs the
frozen reference code path bit for bit.

Reserved fields
---------------
The remaining extension fields, ``eta_Tz``, ``tala`` and an
explicit depth coefficient, exist in the dataclass but are not
implemented.  Setting any of them to a value that would change the physics
raises ``NotImplementedError`` with the message "... not implemented".
``arrhenius_V`` (the pV term) USED to be in that group; the pressure-dependent viscosity law implemented it
under the frozen pV contract (see the "pV contract" section below), so it now
constructs and wires instead of raising.
``dissipation_number`` (the viscous-dissipation number ``Di``) was in that group
until viscous dissipation implemented it: any finite ``Di >= 0`` now constructs and wires the
source ``(Di/Ra) Phi`` of `specmc.physics.dissipation` into the temperature
right-hand side, with ``Di = 0.0``, the frozen reference value, bit for bit
unchanged.  The reserved fields are present so that a scenario file can already
be written and so that the next implementation step has a fixed target; they
are not silently ignored.
"""
from __future__ import annotations

import dataclasses
import math
import warnings
from dataclasses import dataclass, replace
from typing import Any, Dict, Tuple

__all__ = [
    "ConvectionConfig",
    "IMPLEMENTED_VISCOSITY_MODELS", "RESERVED_VISCOSITY_MODELS",
    "IMPLEMENTED_COMPRESSIBILITY", "RESERVED_COMPRESSIBILITY",
    "normalize_viscosity_model", "normalize_compressibility",
    "arrhenius_effective_E",
]

#: viscosity models that `specmc.api.build_model` can build
IMPLEMENTED_VISCOSITY_MODELS = ("const", "eta_z", "eta_T", "arrhenius")
#: viscosity models reserved (raise NotImplementedError)
RESERVED_VISCOSITY_MODELS = ("eta_Tz",)
#: compressibility models that are implemented
IMPLEMENTED_COMPRESSIBILITY = ("boussinesq", "tala")
#: compressibility models reserved (raise NotImplementedError).
#: TALA: the ALA extra pressure term
#: ``(Di/gamma0)(cp0/cv0) rho_bar chi_T p'`` is explicitly out of scope, and a
#: full anelastic (finite Prandtl / dynamic-pressure) formulation with it.
RESERVED_COMPRESSIBILITY = ("ala",)

#: accepted spellings -> canonical model name
_VISC_ALIASES = {
    "const": "const", "constant": "const", "isoviscous": "const",
    "eta_z": "eta_z", "etaz": "eta_z",
    "eta_t": "eta_T", "etat": "eta_T",
    "eta_tz": "eta_Tz", "etatz": "eta_Tz",
    "arrhenius": "arrhenius",
}
_COMP_ALIASES = {
    "boussinesq": "boussinesq", "incompressible": "boussinesq",
    "tala": "tala", "truncated_anelastic": "tala",
    "ala": "ala", "anelastic": "ala",
}


def normalize_viscosity_model(name):
    """Canonicalise a ``viscosity_model`` string (case/alias tolerant).

    Raises ``ValueError`` for a name that does not exist at all; reserved names
    are returned as-is and rejected later by
    :meth:`ConvectionConfig.validate` with ``NotImplementedError``.
    """
    key = str(name).strip().lower()
    if key in _VISC_ALIASES:
        return _VISC_ALIASES[key]
    raise ValueError(
        "unknown viscosity_model %r; implemented: %s; reserved (planned): %s"
        % (name, ", ".join(IMPLEMENTED_VISCOSITY_MODELS),
           ", ".join(RESERVED_VISCOSITY_MODELS)))


def normalize_compressibility(name):
    """Canonicalise a ``compressibility`` string (case/alias tolerant)."""
    key = str(name).strip().lower()
    if key in _COMP_ALIASES:
        return _COMP_ALIASES[key]
    raise ValueError(
        "unknown compressibility %r; implemented: %s; reserved (planned): %s"
        % (name, ", ".join(IMPLEMENTED_COMPRESSIBILITY),
           ", ".join(RESERVED_COMPRESSIBILITY)))


def _impl(name):
    return "%s is not implemented in specmc" % (name,)


def arrhenius_effective_E(cfg):
    """-> the activation ``E`` the config's Arrhenius law will use.

    ``0.0`` when the law is off.  For the T-only law it is
    ``ln(deta_T) * Ts * (1 + Ts)``, the value fixed by the two endpoint
    conditions ``eta(0) = 1`` and ``eta(1) = 1/deta_T``.  For the pV(L)
    branch (``arrhenius_V > 0``) it is the calibrated ``E-tilde`` of
    :func:`specmc.physics.reference_state.calibrate_pv`, which the constants
    contract fixes and ``deta_T`` no longer determines.  Exposed so that a
    driver or an internal check can log the calibrated activation without
    re-deriving it.
    """
    V = float(cfg.arrhenius_V)
    if V > 0.0:
        from .physics.reference_state import (
            REFERENCE_DEFAULTS as _RC, calibrate_pv as _cal)
        a = str(cfg.reference_anchor)
        return float(_cal(_RC, anchor=a, V=V,
                          deta_T=(float(cfg.deta_T) if a == "D"
                                  else None))["E"])
    from .physics.arrhenius import arrhenius_E as _E
    Ts = float(cfg.arrhenius_Ts)
    if Ts <= 0.0:
        return 0.0
    return _E(float(cfg.deta_T), Ts)


@dataclass(frozen=True)
class ConvectionConfig:
    """Physical scenario + scenario-dependent numerical parameters.

    Frozen (immutable) on purpose: a scenario is validated once at construction
    and then handed to a solver that must not be able to change it mid-run.
    Use :meth:`with_overrides` (``dataclasses.replace``) to derive a variant.
    """

    # ======================================================================
    # A. Viscosity (physics)
    # ======================================================================
    #: 'const'   eta = 1, the frozen isoviscous regression baseline
    #: 'eta_z'   eta = exp(-ln(deta_T) * (1 - z)), depth only, frozen
    #: 'eta_T'   eta = exp(-ln(deta_T) * T(x,z,t)), Blankenbach case 2a
    #:           (the reference default)
    #: 'arrhenius'  eta = exp(E/(T+Ts) - E/Ts), the T-only activated-creep law
    #:           (`specmc.physics.arrhenius`); the Arrhenius law, requires
    #:           ``arrhenius_Ts > 0``.  Both laws share the same endpoints
    #:           ``eta(0) = 1``, ``eta(1) = 1/deta_T``.
    viscosity_model: str = "eta_T"
    #: eta(T=0) / eta(T=1) for 'eta_z', 'eta_T' and 'arrhenius'; ignored by
    #: 'const'.
    deta_T: float = 1000.0
    #: reserved: coefficient of an alternative depth law.  Must stay 0.
    eta_z_coeff: float = 0.0
    #: reserved: additional depth dependence.  Must stay 0.
    depth_dependence: float = 0.0
    #: the optional explicit activation of the Arrhenius law,
    #: ``E = ln(deta_T) * Ts * (1 + Ts)``.  ``0.0`` (the default) means
    #: "derive it from the two endpoints", which is the calibrated value; a
    #: nonzero value must EQUAL that value (`validate` rejects anything else),
    #: so this field is a self-documenting echo, never a free parameter.  Must
    #: stay 0 when ``arrhenius_Ts == 0`` (the law is off).
    arrhenius_E: float = 0.0
    #: the Arrhenius law (frozen contract) / the pressure-dependent viscosity law (implemented):
    #: the dimensionless coefficient ``V-tilde`` of the reference
    #: (hydrostatic) pressure term of the full Arrhenius exponent,
    #:
    #:     V-tilde = rho0 * g * d * V_a / (R * Delta T)
    #:     exponent = (E + V-tilde * p0(z)) / (T + Ts) + C ,
    #:                p0(z) = rho0 * g * d * (1 - z)
    #:
    #: ``0.0`` is the default and means "pressure term off": the T-only law of
    #: `specmc.physics.arrhenius` is then exact and bit-for-bit reproducible.
    #: Dynamic-pressure and total-pressure couplings are a DIFFERENT field with
    #: a different numerical cost (an outer eta-p fixed point) and are
    #: explicitly outside this field's meaning.  A strictly positive value
    #: selects the pV(L) branch, whose ``(E-tilde, Ts, C)`` come from
    #: `specmc.physics.reference_state.calibrate_pv` and the eight-constant
    #: `ReferenceConstants` contract; it requires
    #: ``viscosity_model='arrhenius'``, and ``arrhenius_Ts`` / ``arrhenius_E``
    #: must be 0.0 (derive) or equal the calibrated values.  The authoritative
    #: statement is ``specmc.physics.arrhenius.PV_CONTRACT``.
    arrhenius_V: float = 0.0
    #: the pressure-dependent viscosity law: which of the two mutually exclusive calibrations the pV(L)
    #: branch uses (exactly one third condition; see
    #: :func:`specmc.physics.reference_state.calibrate_pv`).
    #:
    #: * ``'D'`` (the default): the frozen reference contrast ``deta_T`` is kept,
    #:   so ``E-tilde = ln(deta_T) Ts (1+Ts) + Ts V-tilde`` and the implied
    #:   ``E_a`` is the output.  This is the only branch consistent with the
    #:   seven-constant contract that was adopted (which does NOT contain
    #:   ``E_a``) and the only one whose contrast is numerically usable: the
    #:   reference-profile range stays ``[1/deta_T, 1]`` and pV *redistributes*
    #:   it with depth (deeper/colder material is stiffer at fixed T), exactly
    #:   the CitcomS ``viscZ`` / ASPECT ``depth prefactor`` pattern.
    #: * ``'Ea'``: ``E-tilde = E_a/(R DeltaT)`` from the constants contract, so
    #:   the contrast is the output, ``D = exp((E-tilde - Ts V-tilde)/
    #:   (Ts (1+Ts))) = e^{193.39}`` with the ASPECT defaults, NOT the
    #:   ``1.1e4`` quoted earlier (that figure came from an
    #:   arithmetic slip; see the `reference_state` module docstring).  This is
    #:   the literal "D is an output" reading, kept for the record.
    #:
    #: Irrelevant (and required to stay at ``'D'``) while ``arrhenius_V == 0``.
    reference_anchor: str = "D"
    #: the Arrhenius temperature offset ``Ts``, in the same units
    #: as ``T`` (so ``T + Ts`` is the absolute-like temperature entering the
    #: activated-creep exponent).  ``0.0``, the default, means "Arrhenius
    #: off": the model then runs the frozen exponential law bit for bit, which
    #: is the reference immune region.  A positive value selects the
    #: Arrhenius law of `specmc.physics.arrhenius` and requires
    #: ``viscosity_model='arrhenius'``.  ``Ts = 1`` is the Frank-Kamenetskii
    #: member (``E = 2 ln deta_T``); the cold-end slope is amplified by
    #: ``(1 + Ts)/Ts`` relative to the exponential law.
    arrhenius_Ts: float = 0.0
    #: explicit eta bounds.  The pointwise viscosity is clipped into
    #: ``[eta_min, eta_max]`` in the padded space, before the truncation back to
    #: the resolved x-modes (`specmc.physics.eta_bounds`).  The default
    #: ``(1e-3, 1e3)`` is the calibration window of the exponential law
    #: (``Delta eta_T = 1000`` either side of the reference viscosity 1); the
    #: frozen case-2a law spans ``[1e-3, 1]``, strictly inside it, so the frozen
    #: reference configuration is bit-for-bit unaffected by the default.
    eta_min: float = 1e-3
    eta_max: float = 1e3
    #: the smooth clip: half-width (in ``ln eta``) of the smooth C^2
    #: transition band that replaces the hard clip near ``eta_min``/``eta_max``
    #: (`specmc.physics.eta_bounds.smooth_clip_eta`).  ``0.0``, the default,
    #: selects the hard clip, bit for bit, which is what keeps the
    #: frozen reference immune region exact.  A positive value must satisfy
    #: ``2*eta_smooth <= ln(eta_max/eta_min)`` so that the two bands cannot meet.
    eta_smooth: float = 0.0
    #: Stokes preconditioner selection.  ``'legacy'`` (the
    #: default) is the frozen reference route, lagged factorisation + adaptive Picard
    #: defect correction, and stays bit for bit.  ``'bfbt'`` selects the
    #: spectral BFBT preconditioned FGMRES route of `specmc.stokes.bfbt` (the
    #: k-block-diagonal Schur-complement model with the harmonic-mean viscosity
    #: profile).  Both routes solve the SAME coupled system; only the Krylov
    #: path differs.
    precond: str = "legacy"

    # ======================================================================
    # B. Heat sources (physics)
    # ======================================================================
    #: internal heating: constant volumetric (internal/radiogenic) heating rate H of
    #: ``d(theta)/dt + u.grad(theta) - u_z = lap(theta) + H``.  Any finite
    #: value is accepted (H > 2 makes the isothermal bottom a net heat sink);
    #: 0.0 is the frozen reference configuration and is bit-for-bit the old path.
    #: See `specmc.physics.heating` for the form, its provenance and the
    #: analytic reference state.
    internal_heating: float = 0.0
    #: the viscous-dissipation number ``Di = alpha0 g0 d / cP0`` of
    #: ``d(theta)/dt + u.grad(theta) - u_z = lap(theta) + H + (Di/Ra) Phi``.
    #: The coefficient is ``Di/Ra``, never 1: the frozen state has
    #: ``<Phi> = Ra(Nu-1) ~ 9e4``, so ``Di = 1`` would drive ``d<T>/dt ~ 9e4``
    #: while ``Di/Ra`` gives ``Di * 9.0``, the physical size of the term
    #: (CitcomS manual :635-640; G-ADOPT approximations.py:400-402).  Any finite
    #: ``Di >= 0`` is legal; ``0.0`` is the frozen reference configuration and is
    #: bit-for-bit the old path.  See `specmc.physics.dissipation`.
    dissipation_number: float = 0.0

    # ======================================================================
    # C. Compressibility (physics)
    # ======================================================================
    #: 'boussinesq' (implemented, the frozen default) | 'tala' (implemented,
    #: TALA) | 'ala' (reserved).  The TALA reference state consumes the SAME
    #: ``dissipation_number`` as the viscous-dissipation source:
    #: ``rho_bar = exp((1-z) Di/gamma)`` with ``gamma = 1`` (``king_ala.prm:46``)
    #: and ``Di = dissipation_number``, so a second field would break the single
    #: source of truth.  The Boussinesq limit is ``Di = 0``, where
    #: ``rho_bar == 1`` exactly and the operator reduces to the frozen one.
    compressibility: str = "boussinesq"

    # ======================================================================
    # D. Physical constants and resolution
    # ======================================================================
    #: Rayleigh number at the reference (maximum) viscosity.
    Ra: float = 1e4
    #: horizontal grid points on the periodic grid of length Lx.
    Nx: int = 64
    #: Chebyshev (CGL) nodes in z.
    Nz: int = 96
    #: length of the periodic carrier grid.  Lx = 2.0 is the 1x1 box carried on
    #: its mirror-doubled grid (see rbc_galerkin.py); it is a property of the
    #: box construction, not a free knob.
    Lx: float = 2.0

    # ======================================================================
    # E. Defect correction (physics/numerics trade-off: spin-up vs developed)
    # ======================================================================
    #: how many steps the frozen factorisation lags the state (16 = spin-up
    #: production value; developed state may use 32).
    lag: int = 16
    #: fixed number of Picard corrections; only used when defect_tol <= 0.
    K: int = 3
    #: adaptive stopping tolerance of the defect correction (relative size of
    #: the last correction).  <= 0 selects the fixed-K route (K above) and is
    #: translated to ``defect_tol=None`` when the model is built.
    defect_tol: float = 1e-8
    #: cap of the adaptive correction count before the correction is declared
    #: failed and the factorisation is refreshed.
    max_K: int = 8

    # ======================================================================
    # F. Symmetry
    # ======================================================================
    #: Mirror (box) projection.  ON is the only validated setting for the
    #: variable-viscosity models: OFF makes the doubled periodic domain a
    #: physical problem whose side conditions are not the 1x1 box.  It is kept
    #: here as a development switch and warns when disabled.
    symmetrize: bool = True

    # ------------------------------------------------------------------ init
    def __post_init__(self):
        self.validate()

    # -------------------------------------------------------------- validate
    def validate(self):
        """Reject reserved/unimplemented features and nonsense values.

        Raises
        ------
        NotImplementedError
            for a reserved extension field set to a value that would change the
            physics (the message always contains "not implemented").
        ValueError
            for a name that does not exist, a non-finite value, or a value that
            cannot be used.
        UserWarning
            (not an error) for ``symmetrize=False`` on a variable-viscosity
            model, which changes the problem definition.

        ``internal_heating`` is NOT rejected any more (internal heating): any finite value
        is a legal scenario, and 0.0 is the frozen reference configuration.
        """
        vm = normalize_viscosity_model(self.viscosity_model)
        if vm in RESERVED_VISCOSITY_MODELS:
            raise NotImplementedError(
                _impl("viscosity_model=%r" % (self.viscosity_model,)))

        comp = normalize_compressibility(self.compressibility)
        if comp in RESERVED_COMPRESSIBILITY:
            raise NotImplementedError(
                _impl("compressibility=%r" % (self.compressibility,)))
        # TALA runs the block-eliminated (reduced) streamfunction
        # operator of `specmc.stokes.stokes_tala`; the 4th-order tau reference
        # operator has no TALA form and is not used by any
        # production route (`_numerics.ZFORM` is the frozen 'reduced').
        if comp == "tala" and str(_numerics().get("zform", "reduced")) != "reduced":
            raise NotImplementedError(
                _impl("compressibility='tala' with zform=%r (only the reduced "
                      "Schur form has a TALA implementation)"
                      % (_numerics().get("zform"),)))
        # TALA: the spectral BFBT preconditioner is built from the frozen
        # Boussinesq block structure (`specmc.stokes.bfbt`); it has no TALA
        # form.  The legacy (lagged defect-correction) route is the
        # TALA production path.
        if comp == "tala" and str(self.precond).lower() == "bfbt":
            raise NotImplementedError(
                _impl("compressibility='tala' with precond='bfbt' (the spectral "
                      "BFBT model is Boussinesq-only; use precond='legacy')"))

        if not math.isfinite(float(self.internal_heating)):
            raise ValueError("internal_heating must be finite (got %r)"
                             % (self.internal_heating,))
        # viscous dissipation: `dissipation_number` is implemented (the `(Di/Ra) Phi` source
        # of `specmc.physics.dissipation`).  Di is a physical dissipation
        # number, so it must be finite and non-negative; a positive value makes
        # the source `(Di/Ra) Phi`, which needs a usable Rayleigh number;
        # the linkage is checked here so the failure names the pair, not only Ra.
        if not math.isfinite(float(self.dissipation_number)):
            raise ValueError("dissipation_number must be finite (got %r)"
                             % (self.dissipation_number,))
        if float(self.dissipation_number) < 0.0:
            raise ValueError(
                "dissipation_number is Di = alpha0 g0 d / cP0 (a physical "
                "dissipation number), so it cannot be negative (got %r)"
                % (self.dissipation_number,))
        if float(self.dissipation_number) > 0.0 and not (
                math.isfinite(float(self.Ra)) and float(self.Ra) > 0.0):
            raise ValueError(
                "dissipation_number=%r makes the source (Di/Ra) Phi, so Ra "
                "must be finite and positive (got Ra=%r)"
                % (self.dissipation_number, self.Ra))

        for name in ("eta_z_coeff", "depth_dependence"):
            if float(getattr(self, name)) != 0.0:
                raise NotImplementedError(
                    _impl("%s=%r" % (name, getattr(self, name))))
        # the Arrhenius law: the temperature offset and activation are now
        # implemented, but ONLY for `viscosity_model='arrhenius'`, the law
        # itself is owned by `specmc.physics.arrhenius`, so the checks that
        # matter live there.  The rules here are the interface ones:
        #   * `arrhenius_Ts <= 0` is "off" (the frozen reference default) and then
        #     `arrhenius_E` must be 0 as well;
        #     anything else would be a knob that silently does nothing;
        #   * `arrhenius_Ts > 0` selects the Arrhenius law, which requires
        #     `viscosity_model='arrhenius'`;
        #   * `arrhenius_E` is an optional explicit activation: 0.0 means
        #     "derive E = ln(deta_T) Ts (1+Ts) from the two endpoints", and a
        #     nonzero value must equal that derived value (the law is
        #     calibrated, not free);
        #   * the pressure-dependent viscosity law: `arrhenius_V > 0` selects the pV(L) branch, for which the
        #     calibration is `reference_state.calibrate_pv`, so the explicit
        #     `arrhenius_Ts`/`arrhenius_E` echoes must match THAT calibration
        #     (both may be left 0.0 to accept the derived values).
        _Ts = float(self.arrhenius_Ts)
        _E = float(self.arrhenius_E)
        _v = float(self.arrhenius_V)
        if not (math.isfinite(_Ts) and math.isfinite(_E)
                and math.isfinite(_v)):
            raise ValueError("arrhenius_E/arrhenius_Ts/arrhenius_V must be "
                             "finite (got %r, %r, %r)"
                             % (self.arrhenius_E, self.arrhenius_Ts,
                                self.arrhenius_V))
        if _Ts < 0.0:
            raise ValueError("arrhenius_Ts is a temperature offset, so it "
                             "cannot be negative (got %r)"
                             % (self.arrhenius_Ts,))
        if _v < 0.0:
            raise ValueError(
                "arrhenius_V is the dimensionless coefficient of the reference "
                "hydrostatic pressure term, V-tilde = rho0 g d V_a/(R Delta T), "
                "so it cannot be negative (got %r)" % (self.arrhenius_V,))
        if _v > 0.0:
            # the pressure-dependent viscosity law: the pV(L) branch.  Ts and E-tilde are the calibration's,
            # not the configuration's; the two echo fields must agree with it or
            # be 0.
            if vm != "arrhenius":
                raise ValueError(
                    "arrhenius_V=%r selects the pV(L) Arrhenius law, which "
                    "needs viscosity_model='arrhenius' (got %r)"
                    % (self.arrhenius_V, self.viscosity_model))
            from .physics.reference_state import (
                REFERENCE_DEFAULTS as _RC, PV_ANCHORS as _ANCH,
                calibrate_pv as _cal)
            _anchor = str(self.reference_anchor)
            if _anchor not in _ANCH:
                raise ValueError(
                    "unknown reference_anchor %r; use one of %r"
                    % (self.reference_anchor, _ANCH))
            _rec = _cal(_RC, anchor=_anchor, V=_v,
                        deta_T=(float(self.deta_T) if _anchor == "D" else None))
            if _Ts != 0.0 and abs(_Ts - _rec["Ts"]) > 1e-12 * _rec["Ts"]:
                raise ValueError(
                    "arrhenius_Ts=%r contradicts the reference-state value "
                    "Ts = T_surf/DeltaT = %r; under pV(L) Ts is fixed by the "
                    "constants contract (leave it 0.0 to accept the derived "
                    "value)" % (self.arrhenius_Ts, _rec["Ts"]))
            if _E != 0.0 and abs(_E - _rec["E"]) > 1e-12 * abs(_rec["E"]):
                raise ValueError(
                    "arrhenius_E=%r contradicts the calibrated pV activation "
                    "E-tilde = %r (V-tilde=%r, anchor=%r); leave it 0.0 to "
                    "accept the derived value"
                    % (self.arrhenius_E, _rec["E"], _v, _anchor))
        elif str(self.reference_anchor) != "D":
            raise ValueError(
                "reference_anchor=%r has no effect while arrhenius_V == 0 "
                "(the pV term is off), so it must stay at its default 'D'; a "
                "silently ignored knob is not allowed"
                % (self.reference_anchor,))
        elif _Ts == 0.0 and _E != 0.0:
            raise ValueError(
                "arrhenius_E=%r is set but arrhenius_Ts is 0, which means "
                "'Arrhenius off'; the activation would be silently ignored.  "
                "Use viscosity_model='arrhenius' with a positive "
                "arrhenius_Ts." % (self.arrhenius_E,))
        elif _Ts > 0.0:
            if vm != "arrhenius":
                raise ValueError(
                    "arrhenius_Ts=%r selects the Arrhenius law, which needs "
                    "viscosity_model='arrhenius' (got %r)"
                    % (self.arrhenius_Ts, self.viscosity_model))
            from .physics.arrhenius import arrhenius_E as _derived_E
            _E_ref = _derived_E(float(self.deta_T), _Ts)
            if _E != 0.0 and abs(_E - _E_ref) > 1e-12 * abs(_E_ref):
                raise ValueError(
                    "arrhenius_E=%r contradicts the calibrated value "
                    "E = ln(deta_T) Ts (1+Ts) = %r for deta_T=%r, Ts=%r; the "
                    "law is calibrated by the two endpoints, so E is not a "
                    "free parameter" % (self.arrhenius_E, _E_ref, self.deta_T,
                                        self.arrhenius_Ts))
        # the pressure-dependent viscosity law: `arrhenius_V` is now implemented (the pV(L) branch is validated
        # in the block above, where it shares the calibration with the model).
        # the viscosity bounds are implemented, so validate them instead
        # of rejecting them.  `specmc.physics.eta_bounds` owns the rule.
        from .physics.eta_bounds import validate_bounds, validate_smooth
        lo, hi = validate_bounds(self.eta_min, self.eta_max)
        # the smooth clip: the smoothing width is validated against the window
        # that was accepted (0.0 is the hard clip).
        validate_smooth(self.eta_smooth, lo, hi)
        # the preconditioner selector.  A typo must
        # not be silently ignored (it is an error, not a no-op).
        if str(self.precond).strip().lower() not in ("legacy", "bfbt"):
            raise ValueError(
                "unknown precond %r; use 'legacy' (the frozen reference defect-"
                "corrected route) or 'bfbt' (the spectral BFBT preconditioned "
                "FGMRES route)" % (self.precond,))
        if vm == "const" and not (lo <= 1.0 <= hi):
            raise ValueError(
                "viscosity_model='const' has eta == 1 everywhere, so the "
                "requested eta window [%r, %r] would clip the whole field; "
                "use 'eta_z' or 'eta_T' for a bounded rheology" % (lo, hi))

        if float(self.Ra) <= 0.0:
            raise ValueError("Ra must be positive (got %r)" % (self.Ra,))
        if int(self.Nx) != self.Nx or int(self.Nx) < 4:
            raise ValueError("Nx must be an integer >= 4 (got %r)"
                             % (self.Nx,))
        if int(self.Nz) != self.Nz or int(self.Nz) < 4:
            raise ValueError("Nz must be an integer >= 4 (got %r)"
                             % (self.Nz,))
        if float(self.Lx) <= 0.0:
            raise ValueError("Lx must be positive (got %r)" % (self.Lx,))
        if float(self.deta_T) <= 0.0:
            raise ValueError("deta_T must be positive (got %r)"
                             % (self.deta_T,))
        if int(self.lag) != self.lag or int(self.lag) < 1:
            raise ValueError("lag must be an integer >= 1 (got %r)"
                             % (self.lag,))
        if int(self.max_K) != self.max_K or int(self.max_K) < 1:
            raise ValueError("max_K must be an integer >= 1 (got %r)"
                             % (self.max_K,))
        if int(self.K) != self.K or int(self.K) < 0:
            raise ValueError("K must be an integer >= 0 (got %r)" % (self.K,))
        if float(self.defect_tol) <= 0.0 and int(self.K) < 1:
            raise ValueError("defect_tol <= 0 selects the fixed-K route, so K "
                             "must be >= 1 (got K=%r)" % (self.K,))
        if self.K > self.max_K:
            raise ValueError("K=%r must not exceed max_K=%r"
                             % (self.K, self.max_K))

        if vm != "const" and not bool(self.symmetrize):
            warnings.warn(
                "symmetrize=False on viscosity_model=%r removes the box "
                "projection: the doubled periodic domain becomes the physical "
                "problem and the 1x1 free-slip/insulating side conditions no "
                "longer hold.  This is a development setting, not a physical "
                "scenario." % (self.viscosity_model,), UserWarning, stacklevel=2)

    # ---------------------------------------------------------------- helpers
    def canonical(self) -> Dict[str, Any]:
        """-> field values with the string switches canonicalised."""
        d = self.as_dict()
        d["viscosity_model"] = normalize_viscosity_model(self.viscosity_model)
        d["compressibility"] = normalize_compressibility(self.compressibility)
        return d

    def as_dict(self) -> Dict[str, Any]:
        """-> every field as a plain dict (construction order preserved)."""
        return {f.name: getattr(self, f.name)
                for f in dataclasses.fields(self)}

    def with_overrides(self, **kwargs) -> "ConvectionConfig":
        """-> a new validated config with ``kwargs`` replaced."""
        return replace(self, **kwargs)

    @classmethod
    def field_names(cls) -> Tuple[str, ...]:
        return tuple(f.name for f in dataclasses.fields(cls))


def _numerics():
    # local import: keeps `specmc.config` importable even while `_numerics`
    # bootstraps sys.path, and avoids a circular import at module load.
    from . import _numerics as num
    return num.as_dict()
