"""Viscosity (rheology) laws for the 2D infinite-Prandtl convection model.

PROVENANCE POLICY
-----------------
Every law in this file is a transcription of a formula that is published or
implemented in an authoritative code.  Nothing here is invented.  For each law
this file records

  * the exact expression,
  * the source (paper / official documentation / official source file),
  * the corresponding implementation site in ASPECT and CitcomS,
  * the parameter mapping between the two symbol sets.

The project convention (identical to `rbc_galerkin.py`) is

    T(x,z,t) = Tbar(z) + theta(x,z,t),     Tbar(z) = 1 - z   (z = 0 is the hot
                                                              bottom, z = 1 is
                                                              the cold top)

and the Stokes equation is written with the Rayleigh number *explicit* in the
buoyancy term and a *relative* viscosity field eta(x,z):

    -div[ eta (grad u + grad u^T) ] + grad p = Ra * theta * e_z

so eta == 1 is the isoviscous reference state that defines Ra.  A constant
rescaling eta -> c*eta is exactly equivalent to Ra -> Ra/c (verified with
an independent longhand cross-check).


Law 1: isoviscous
    eta(T) = 1
    Source: Blankenbach et al. (1989) case 1a (Delta eta_T = 1); this is the
    frozen regression baseline of the project.

Law 2: exponential temperature dependence ("Blankenbach case 2a")

    eta(T) = exp( -ln(Delta eta_T) * T ) ,       T in [0, 1]
    Delta eta_T = eta(T=0) / eta(T=1) = eta_cold / eta_hot

    Source (paper):    Blankenbach, Busse, Christensen, Cserepes, Gunkel,
        Hansen, Harder, Jarvis, Koch, Marquart, Moore, Olson, Schmeling &
        Schnaubelt (1989), "A benchmark comparison for mantle convection
        codes", Geophys. J. Int. 98, 23-38; case 2a, Table 9.
        Reference values used here:  Nu = 10.066, Vrms = 480.4334
        (verbatim copy of ASPECT's curated table,
         benchmarks/blankenbach/reference_statistics.txt).

    Source (ASPECT, the run that reproduces those numbers):
        benchmarks/blankenbach/base_case2a.prm
            set Ra                              = 1e4
            set Viscosity temperature prefactor = 6.907755279   (= ln 1000)
            set Viscosity depth prefactor       = 0.0
        source/material_model/nondimensional.cc:
            out.viscosities[i] = (compressible ? Di : 1.0)/Ra
                                 * exp(-exponential_viscosity_temperature_prefactor
                                       * temperature_deviation
                                       + exponential_viscosity_depth_prefactor
                                       * depth);
        i.e.  eta_ASPECT = (1/Ra) * exp(-b * T') with T' = T - T_adiabatic.
        ASPECT's case 2a uses the `function` adiabatic conditions
        ("0; 0; 1", i.e. T_adiabatic = 0), so T' = T = full temperature.
        ASPECT puts the 1/Ra *inside* the viscosity while its buoyancy term is
        O(1) (gravity magnitude 1, density = 1 - T).  The convention here is the
        mirror image (buoyancy = Ra*theta, viscosity relative).  The two are
        the same equations: removing ASPECT's constant 1/Ra prefactor gives
        exactly law 2 above.
        IMPORTANT: because of the 1/Ra, ASPECT's Ra = 1e4 is the Rayleigh
        number of the *stiffest* (coldest) material; the hot bottom has a
        local Ra of Ra * Delta eta_T = 1e7, which is why Vrms jumps from 42.86
        (case 1a) to 480.43 (case 2a) at the same nominal Ra.

    Source (ASPECT, alternative plugin, the adopted mapping):
        source/material_model/rheology/frank_kamenetskii.cc:
            viscosity = prefactor
                        * exp(viscosity_ratio * 0.5 * (1.0 - T/reference_temperature))
        Matching the ratio eta(T=0)/eta(T=T_ref) = Delta eta_T requires
            E_FK = 2 * ln(Delta eta_T),      prefactor = eta_ref.
        With Delta eta_T = 1000 this gives E_FK = 13.815510558 = 2*ln(1000),
        i.e. exactly twice the number that appears in ASPECT's own case-2a
        parameter file (6.907755279) because of the 0.5 factor.

    Source (CitcomS):
        lib/Viscosity_structures.c, switch(E->viscosity.RHEOL):
          case 1:  eta = N0 * exp( viscE * (viscT - T) )
          case 6:  eta = N0 * exp( viscE * (viscT - T) + (1 - z) * viscZ )
                   ("like case 1, but allowing for depth-dependence if
                     Z_0 != 0")
        Parameter mapping for law 2 (pure temperature dependence):
            N0    = 1.0            (visc0)
            viscE = ln(Delta eta_T)      (viscE; = 6.907755279 for 1000)
            viscT = 0.0            (viscT)
            viscZ = 0.0            (viscZ, case 6 only)
        Then eta = exp(ln(Delta eta_T)*(0 - T)) = exp(-ln(Delta eta_T) T).

    Source (Underworld2 official benchmark notebook):
        docs/examples/1_04_BlankenbachBenchmark_Case2a.ipynb
            eta0 = 1.0e3;  b = math.log(eta0)
            fn_viscosity = eta0 * fn.math.exp(-1.0 * b * T)
        i.e. eta = Delta eta_T * exp(-ln(Delta eta_T) T), the same law up to
        the overall factor Delta eta_T (a constant rescaling of viscosity).
        Underworld quotes Ra = 1e7 for this case because it absorbs the
        prefactor into Ra instead of into the viscosity normalization.

Depth-only usage
    eta(z) = exp( -ln(Delta eta_T) * Tbar(z) ),   Tbar(z) = 1 - z
i.e. law 2 evaluated on the *conductive background* temperature profile.
This is a deliberate simplification (the frozen specification): the
viscosity is frozen to a function of depth, so the Stokes operator is
time-independent and the per-wavenumber LU cache survives.  At Delta eta_T = 1
it is exactly the isoviscous baseline.  It is *not* the case-2a physics (which
needs eta = eta(T(x,z,t)) with the evolving temperature, the coupled problem).

Law 3: Arrhenius (T-only)
    eta(T) = exp( E/(T + Ts) - E/Ts ),   E = ln(Delta eta_T) * Ts * (1 + Ts)

is NOT implemented here.  Its single source of truth is
:mod:`specmc.physics.arrhenius`; this module only records the relationship,
because the two laws are easy to confuse:

  * they share the endpoints ``eta(0) = 1`` and ``eta(1) = 1/Delta eta_T`` for
    every ``Ts > 0``, so a two-point check CANNOT tell them apart;
  * they differ in the profile: ``Ts = 1`` gives ``E = 2 ln(Delta eta_T)`` (the
    Frank-Kamenetskii activation recorded above) but ``ln eta(0.5)`` differs
    from law 2 by ``1.151`` (a factor 3.16 in eta);
  * law 2 is the ``Ts -> inf`` limit of law 3, with an ``O(ln(D)^2/Ts)`` gap,
    i.e. law 2 is a linearisation of law 3, not a member of its family;
  * the cold-end slope of law 3 is ``ln(D) * (1 + Ts)/Ts``, i.e. ``2x`` the
    constant ``ln D`` of law 2 at ``Ts = 1``.

Its single source of truth and its calibration are in
:mod:`specmc.physics.arrhenius`.
"""
import numpy as np

__all__ = [
    "eta_exp_T", "eta_conductive_background", "eta_from_law",
    "LAW_PARAMETER_MAPPING", "reference_values",
]

# Recorded in one place so every call site quotes the same numbers.
# Source: ASPECT benchmarks/blankenbach/reference_statistics.txt, which is a
# transcription of Blankenbach et al. (1989) Table 9.
REFERENCE = {
    "1a": dict(Ra=1e4, deta_T=1.0, Nu=4.884409, Vrms=42.864947),
    "1b": dict(Ra=1e5, deta_T=1.0, Nu=10.534095, Vrms=193.21454),
    "1c": dict(Ra=1e6, deta_T=1.0, Nu=21.972465, Vrms=833.98977),
    "2a": dict(Ra=1e4, deta_T=1000.0, Nu=10.066, Vrms=480.4334),
    "2b": dict(Ra=1e5, deta_T=1000.0, Nu=6.9299, Vrms=171.755),
}

# The frozen isoviscous regression target, reproduced
# bit-for-bit by the reference regression run).
BASELINE_REF = dict(Ra=1e4, Nx=64, Nz=48, t=0.25,
                    Nu=4.885140, Vrms=42.872720)


def eta_exp_T(T, deta_T):
    """Law 2:  eta(T) = exp(-ln(Delta eta_T) * T).

    Blankenbach et al. (1989) case 2a, as implemented in ASPECT
    (`nondimensional` material model, `Viscosity temperature prefactor`
    = ln(Delta eta_T)) and in CitcomS (RHEOL=1, viscE = ln(Delta eta_T),
    viscT = 0).
    """
    return np.exp(-np.log(float(deta_T)) * np.asarray(T, dtype=float))


def eta_conductive_background(z, deta_T, Tbot=1.0, Ttop=0.0):
    """Depth-only viscosity: law 2 on Tbar(z) = Tbot + (Ttop-Tbot) z.

    z = 0 is the hot bottom (Tbar = 1 -> eta = 1/Delta eta_T, weak),
    z = 1 is the cold top   (Tbar = 0 -> eta = 1, stiff, = the reference
    viscosity that defines Ra).  Same sense as ASPECT's case 2a, where the
    stiffest material carries the nominal Ra.
    """
    T = Tbot + (Ttop - Tbot) * np.asarray(z, dtype=float)
    return eta_exp_T(T, deta_T)


def eta_from_law(law, deta_T, z):
    """Dispatch table for the model's `viscosity_law` switch."""
    law = str(law).lower()
    if law in ("isoviscous", "const", "constant"):
        return np.ones_like(np.asarray(z, dtype=float))
    if law in ("exp_t_depth", "exp_T_depth", "depth_exp"):
        return eta_conductive_background(z, deta_T)
    raise NotImplementedError(
        "viscosity_law %r is not implemented.  The depth-only viscosity supports "
        "'isoviscous' and 'exp_T_depth'." % (law,))


LAW_PARAMETER_MAPPING = {
    # symbol here           ASPECT (nondimensional / FrankKamenetskii)   CitcomS
    "Delta eta_T": dict(
        aspect_nondim="exp(Viscosity temperature prefactor)",
        aspect_fk="exp(E_FK/2)",
        citcoms="exp(viscE) (with viscT = 0)"),
    "ln(Delta eta_T)": dict(
        aspect_nondim="Viscosity temperature prefactor  (= 6.907755279 for 1000)",
        aspect_fk="E_FK / 2",
        citcoms="viscE"),
    "E (Frank-Kamenetskii)": dict(
        aspect_nondim="2 * Viscosity temperature prefactor",
        aspect_fk="E_FK  (= 2 ln(Delta eta_T))",
        citcoms="2 * viscE"),
    "reference temperature": dict(
        aspect_nondim="adiabatic surface temperature (case 2a: T_adia = 0)",
        aspect_fk="reference_temperature (T_ref)",
        citcoms="viscT (= 0)"),
    "depth prefactor": dict(
        aspect_nondim="Viscosity depth prefactor c (case 2a: 0)",
        aspect_fk="not available",
        citcoms="viscZ (RHEOL=6; case 2a: 0)"),
    "Ra convention": dict(
        aspect_nondim="Ra = 1e4 is the Ra of the STIFFEST material "
                      "(eta = 1/Ra inside the model)",
        aspect_fk="n/a",
        citcoms="Ra = 1e4 at the reference (maximum) viscosity"),
}


def reference_values(case):
    """Benchmark reference values, transcribed from the sources above."""
    return dict(REFERENCE[str(case).lower()])
