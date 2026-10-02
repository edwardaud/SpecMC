"""Internal (volumetric) heating for the 2D infinite-Prandtl convection model.

What is implemented, and why in this form
=========================================
A *constant* volumetric heat production H is added to the temperature equation
of `specmc`.  With the project's non-dimensionalisation (unit box height, unit
thermal diffusivity, Delta T = 1, unit thermal conductivity, z = 0 the hot
bottom) the energy equation is

    dT/dt + u.grad(T) = lap(T) + H ,        T = 1 at z = 0 ,  T = 0 at z = 1

and, because the project writes T = (1 - z) + theta,

    d(theta)/dt + u.grad(theta) - u_z = lap(theta) + H .

The source is a plain, un-scaled additive constant:
* it does not carry a Rayleigh factor (the internal-heating Rayleigh number
  ``Ra_H = Ra H`` is a re-parameterisation of the drive, not a source term);
* it is *not* divided by Ra (that only happens in the non-standard
  ``(1/Ra) lap(T)`` normalisation, which no reference code uses);
* the background-advection term ``-u_z`` is untouched, because
  ``lap(1 - z) = 0`` and the source does not modify the background profile.

PROVENANCE (the same policy as `specmc.physics.viscosity`)
----------------------------------------------------------
Four independent, mature mantle-convection codes implement exactly this source;
none of them uses ``Ra H`` or ``H/Ra`` as the additive term.

ASPECT (deal.II).  Manual, "Basic equations", temperature equation:

    rho C_p (dT/dt + u.grad T) - div(k grad T)
        = rho H + 2 eta (...) : (...) + alpha T (u.grad p) + ... (latent)

  The heating models are summed source terms; the constant one is

    Heating model / Constant heating / Radiogenic heating rate   [W/kg]

  documented verbatim as: "This parameter corresponds to the variable H in the
  temperature equation stated in the manual, and the heating term is rho H."
  (source/heating_model/constant_heating.cc:
  ``heating_source_terms[q] = radiogenic_heating_rate * densities[q]``).
  With rho = 1 this is ``+H``.
  URL: https://aspect-documentation.readthedocs.io/en/stable/user/methods/basic-equations/index.html
       https://github.com/geodynamics/aspect/blob/main/source/heating_model/constant_heating.cc

CitcomS (CIG).  Manual, dimensionless energy equation (Boussinesq limit):

    rho c_P (T_,t + u_i T_,i) = rho c_P kappa T_,ii + (Di/Ra) Phi + rho H

  Input keyword ``Q0`` ("this specifies the internal heating number",
  lib/Instructions.c), and lib/Advection_diffusion.c builds
  ``heating = rho * Q0``.  With rho = 1 this is ``+Q0`` = ``+H``.
  URL: https://github.com/geodynamics/citcoms/blob/master/doc/citcoms-manual.pdf
       https://github.com/geodynamics/citcoms/blob/master/lib/Advection_diffusion.c

Underworld.  Underworld2 `AdvectionDiffusion` docstring gives
``dphi/dt + u.grad phi = div(k grad phi) + H`` with the source supplied through
``fn_sourceTerm`` (steady state: ``fn_heating``); UWGeodynamics names the
material property ``radiogenicHeatProd`` [W/m^3] and maps it to
``H / (rho c_p)``.  Underworld3's internally heated example sets
``adv_diff.f = H_int``.  With rho = c_p = 1 all of these are ``+H``.
  URL: https://github.com/underworldcode/underworld2/blob/master/src/underworld/systems/_advectiondiffusion.py
       https://github.com/underworldcode/underworld3/blob/main/docs/examples/convection/advanced/Ex_Convection_Disc_InternalHeat.py

G-ADOPT (Firedrake).  ``BoussinesqApproximation(Ra, ..., H=...)`` with
``energy_source = rho * H`` (``ExtendedBoussinesqApproximation`` adds viscous
dissipation to the same term); the energy solver assembles
``mass + advection + diffusion = source``.  Again ``+rho H`` = ``+H`` at
rho = 1, and Ra appears only in the buoyancy.
  URL: https://github.com/g-adopt/g-adopt/blob/main/gadopt/approximations.py
       https://gadopt.org/documentation/api/approximations/

The spectral argument.  H is constant, so in the Fourier(x) x Chebyshev(z)
discretisation it occupies exactly ONE coefficient: the T_0 Chebyshev
coefficient of the k_x = 0 Fourier mode.  It is therefore injected without any
dealiasing error and without touching any other coefficient, see
:func:`add_constant_source`.

What H does NOT include
-----------------------
Viscous dissipation (`dissipation_number`) is a separate source, implemented by
:mod:`specmc.physics.dissipation`: ``(Di/Ra) Phi`` with
``Phi = 2 eta (eps - 1/3 tr(eps) 1) : eps``.  This module implements *bulk*
heating only, and :func:`balance` is the ``H``-only budget; its ``H + Phi``
generalisation is ``specmc.physics.dissipation.balance``, which CALLS
:func:`balance` and subtracts the dissipation term from the residual.

Analytic reference state
========================
The motionless steady solution of ``d2T/dz2 + H = 0`` with T(0) = 1, T(1) = 0
is a quadratic, hence exactly representable in the Chebyshev basis:

    T_cond(z)     = (1 - z) + (H/2) z (1 - z)
    theta_cond(z) =           (H/2) z (1 - z)

and therefore

    nu_bot = 1 - H/2 ,   nu_top = 1 + H/2 ,   <T> = 1/2 + H/12 ,
    interior maximum  T_max = 1/2 + H/8 + 1/(2H)  at  z = 1/2 - 1/H   (H > 2).

For H > 2 the isothermal bottom turns into a net heat *sink* (Nu_bot < 0): the
conductive profile is hottest inside the layer.  That is a property of the
scenario, not of the discretisation.

Energy identity
===============
Integrating the equation over the unit box (incompressible, u.n = 0, insulating
side walls) gives the exact budget

    d<T>/dt = Nu_bot - Nu_top + H          (unit volume, unit conductivity),

so at a steady state ``Nu_top - Nu_bot = H``: whatever is produced inside must
leave through the top.  Horizontally averaged, the total upward flux is

    q(z) = conj(u_z T)(z) - dTbar/dz = Nu_bot + H z .

:func:`balance` evaluates the budget's four terms (and its residual) from a
model state; :func:`nu_energy_steady` gives the H-corrected version of the
frozen ``Nu_energy = 1 + <u_z theta>`` identity, which becomes

    Nu_bot = 1 + <u_z theta> - H/2 ,   Nu_top = 1 + <u_z theta> + H/2

(so the *frozen* ``Nu_energy`` is NOT the H-corrected one when H != 0).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "PARAMETER_MAPPING", "conductive_theta", "conductive_T", "mean_T_conductive",
    "nu_bot_conductive", "nu_top_conductive", "T_max_conductive",
    "source_coeffs", "add_constant_source", "source_integral",
    "nu_energy_steady", "nu_bot_energy_steady", "balance", "flux_profile",
    "describe", "reference_values", "self_test",
]

#: how the same constant is spelled in the codes this form was chosen from
PARAMETER_MAPPING = {
    "ASPECT": dict(
        path="Heating model / Constant heating / Radiogenic heating rate",
        units="W/kg", term="rho * H  (source/heating_model/constant_heating.cc)",
        note="rho = 1 in a non-dimensional run -> the input value IS H"),
    "CitcomS": dict(
        path="input file keyword Q0", units="-", term="rho * Q0",
        note="lib/Instructions.c parses Q0; lib/Advection_diffusion.c uses "
             "`heating = rho * Q0`.  Ra_H = Ra*H is a derived number, not "
             "the source."),
    "Underworld2": dict(
        path="AdvectionDiffusion(fn_sourceTerm=...) / SteadyStateHeat("
             "fn_heating=...) / UWGeodynamics material.radiogenicHeatProd",
        units="W/m^3 (UWGeodynamics only)", term="H  (or H/(rho c_p))",
        note="UWGeodynamics converts the dimensional W/m^3 value to the "
             "non-dimensional source H/(rho c_p)"),
    "Underworld3": dict(
        path="AdvDiffusionSLCN.f", units="-", term="H",
        note="docs/examples/convection/advanced/Ex_Convection_Disc_InternalHeat.py"),
    "G-ADOPT": dict(
        path="BoussinesqApproximation(Ra, ..., H=H)", units="-",
        term="rho * H  (approximations.py::energy_source)",
        note="Ra appears only in the buoyancy term"),
}
SECTION = "internal heating"


# --------------------------------------------------------------------------
# analytic reference state
# --------------------------------------------------------------------------
def conductive_theta(z, H):
    """theta = T - (1 - z) of the motionless steady state with constant H."""
    z = np.asarray(z, dtype=float)
    return 0.5 * float(H) * z * (1.0 - z)


def conductive_T(z, H):
    """The analytic purely-conductive temperature profile (exact quadratic)."""
    z = np.asarray(z, dtype=float)
    return (1.0 - z) + conductive_theta(z, H)


def mean_T_conductive(H):
    """<T> of the conductive state: 1/2 + H/12."""
    return 0.5 + float(H) / 12.0


def nu_bot_conductive(H):
    """Nu_bot = 1 - H/2 (negative for H > 2: the bottom becomes a sink)."""
    return 1.0 - float(H) / 2.0


def nu_top_conductive(H):
    """Nu_top = 1 + H/2."""
    return 1.0 + float(H) / 2.0


def T_max_conductive(H):
    """-> (T_max, z_max) of the conductive profile (H > 2 puts it inside)."""
    H = float(H)
    if H <= 2.0:
        return 1.0, 0.0
    return 0.5 + H / 8.0 + 0.5 / H, 0.5 - 1.0 / H


# --------------------------------------------------------------------------
# the discrete source
# --------------------------------------------------------------------------
def source_coeffs(Nk, Nz, H):
    """-> the (Nk, Nz) spectral representation of ``theta_t += H``.

    A constant has exactly one non-zero Fourier/Chebyshev coefficient: the
    Chebyshev T_0 coefficient of the k_x = 0 mode (``V @ c`` is the value
    vector, and ``T_0 = 1`` everywhere).
    """
    out = np.zeros((int(Nk), int(Nz)), dtype=complex)
    if float(H) != 0.0:
        out[0, 0] = complex(float(H))
    return out


def add_constant_source(b, Nk, Nz, H):
    """-> ``b`` with the internal-heating coefficient added (H == 0: unchanged).

    ``b`` is the raveled ``(Nk, Nz)`` right-hand side of the theta equation.
    A zero source returns the same object, so the frozen H = 0 code path
    is bit-for-bit what it was (the immune region).
    """
    if float(H) == 0.0:
        return b
    out = np.array(b, copy=True).reshape(int(Nk), int(Nz))
    out[0, 0] += float(H)
    return out.ravel()


def source_integral(H):
    """-> the exact volume integral of the constant source over the unit box.

    ``wq . (H e_0) = H * gk_0 = H`` exactly (``gk_0 = 1``), which is what makes
    the source's contribution to the discrete energy budget exact.
    """
    return float(H)


# --------------------------------------------------------------------------
# energy bookkeeping
# --------------------------------------------------------------------------
def nu_energy_steady(uz_theta, H=0.0):
    """-> 1 + <u_z theta>, the value SHARED by Nu_bot and Nu_top at a steady
    state (see :func:`nu_bot_energy_steady` / +H/2 for the top)."""
    return 1.0 + float(uz_theta)


def nu_bot_energy_steady(uz_theta, H=0.0):
    """-> Nu_bot implied by the steady energy identity: 1 + <u_z th> - H/2."""
    return 1.0 + float(uz_theta) - 0.5 * float(H)


def flux_profile(model, that, H=0.0):
    """-> (z, q(z)) with ``q = conj(u_z T) - dTbar/dz = Nu_bot + H z``.

    The identity ``q(z) = Nu_bot + H z`` is the horizontally averaged energy
    equation; it is the H-generalisation of "the total flux is height
    independent".
    """
    that = np.asarray(that)
    psi, _ = model.stokes(that)
    ux, uz = model.velocity(psi)
    Tf = 1.0 - model.z[None, :] + np.real(model.grid(that))
    uzT = (uz * Tf).mean(axis=0)
    # the x-mean of T is 1 - z + theta_0(z), so its z-derivative at the nodes
    # is -1 + (V @ (D @ that[0])).real
    dTbar = -1.0 + (model.V @ (model.D @ that[0])).real
    return model.z, uzT - dTbar


def _nu_pair(model, that):
    dth = model.D @ np.asarray(that)[0]
    return (float(1.0 - (model.V[0] @ dth).real),
            float(1.0 - (model.V[-1] @ dth).real))


def _theta_mean(model, that):
    return float(model.vol_mean(model.grid(that)))


def balance(model, that_np1, that_n, that_nm1, a0, b0, b1, H=0.0,
            advection=None):
    """-> the four terms (and residual) of the discrete energy budget.

    storage   = a0<theta^{n+1}> - b0<theta^n> + b1<theta^{n-1}>
    nu_bot, nu_top   boundary fluxes of theta^{n+1}
    heating   = H                      (the source's exact volume integral)
    residual  = storage - (nu_bot - nu_top) - heating - advection

    ``advection`` is ``<u* . grad theta^{n+1}>`` as the scheme
    computes it (0 for the exact operator; pass
    ``model.vol_mean(model.grid(model.theta.advect(that_np1)))`` for the
    discrete value).  The *enforced* (tau) rows of the balance close to machine
    precision; the residual measures the two un-enforced tau rows.
    """
    nu_bot, nu_top = _nu_pair(model, that_np1)
    storage = (a0 * _theta_mean(model, that_np1)
               - b0 * _theta_mean(model, that_n)
               + (b1 * _theta_mean(model, that_nm1)
                  if that_nm1 is not None else 0.0))
    adv = 0.0 if advection is None else float(advection)
    heating = source_integral(H)
    resid = storage - (nu_bot - nu_top) - heating - adv
    return dict(storage=storage, nu_bot=nu_bot, nu_top=nu_top,
                heating=heating, advection=adv, residual=resid,
                theta_mean=_theta_mean(model, that_np1),
                T_mean=0.5 + _theta_mean(model, that_np1))


def describe(H=0.0):
    """-> a JSON-safe description of the active internal-heating setting."""
    H = float(H)
    return dict(section=SECTION, internal_heating=H, enabled=bool(H != 0.0),
                equation="d(theta)/dt + u.grad(theta) - u_z = lap(theta) + H",
                source_coefficient="(k_x = 0, T_0 Chebyshev)",
                parameter_mapping=dict(PARAMETER_MAPPING),
                conductive=dict(theta_max=float(conductive_theta(0.5, H)),
                                T_mean=mean_T_conductive(H),
                                nu_bot=nu_bot_conductive(H),
                                nu_top=nu_top_conductive(H)))


def reference_values(H=0.0):
    """-> the analytic conduction reference state for ``H`` (a fresh dict)."""
    Tmax, zmax = T_max_conductive(H)
    return dict(H=float(H), theta_max=float(conductive_theta(0.5, H)),
                T_mean=mean_T_conductive(H), nu_bot=nu_bot_conductive(H),
                nu_top=nu_top_conductive(H), T_max=Tmax, z_max=zmax)


# --------------------------------------------------------------------------
# self test (runs in seconds; wired into the package self-tests)
# --------------------------------------------------------------------------
def self_test(verbose=True):
    """Structural checks of the module's algebra (no model, no external
    driver)."""
    ok = True
    H = 10.0
    # [1] the analytic state is exactly representable and self-consistent:
    #     theta'' = -H exactly, and -dT/dz at the two walls is nu_bot/nu_top
    z = np.linspace(0.0, 1.0, 201)
    th = conductive_theta(z, H)
    q = np.polyfit(z, th, 2)
    e_quad = abs(2 * q[0] + H)                       # theta'' = 2*a = -H
    dz = 1e-9
    dT_bot = (conductive_T(dz, H) - conductive_T(0.0, H)) / dz
    dT_top = (conductive_T(1.0, H) - conductive_T(1.0 - dz, H)) / dz
    e_bot = abs(-dT_bot - nu_bot_conductive(H))
    e_top = abs(-dT_top - nu_top_conductive(H))
    ok &= (e_quad < 1e-9) and (e_bot < 1e-6) and (e_top < 1e-6)
    if verbose:
        print('  [1] conduction: |theta_zz + H| = %.1e, |Nu_bot - %.6f| = '
              '%.1e, |Nu_top - %.6f| = %.1e'
              % (e_quad, nu_bot_conductive(H), e_bot, nu_top_conductive(H),
                 e_top))
    # [2] the integral of theta_cond over [0,1] is H/12 (so <T> = 1/2 + H/12)
    w = np.polynomial.legendre.leggauss(64)
    zz, ww = 0.5 * (w[0] + 1.0), 0.5 * w[1]
    e_int = abs(np.sum(ww * conductive_theta(zz, H)) - H / 12.0)
    ok &= e_int < 1e-14
    if verbose:
        print('  [2] int_0^1 theta_cond dz = H/12: err %.1e' % (e_int,))
    # [3] the mean-T / flux identity: <T> = 1/2 + H/12, nu_bot - nu_top = -H
    e_id = abs((nu_bot_conductive(H) - nu_top_conductive(H)) + H)
    ok &= e_id < 1e-15
    if verbose:
        print('  [3] nu_bot - nu_top + H = %.1e   <T> err %.1e'
              % (e_id, abs(mean_T_conductive(H) - (0.5 + H / 12.0))))
    # [4] H = 0 is a hard no-op: add_constant_source returns the SAME object
    b = np.arange(12.0).reshape(3, 4).astype(complex)
    b0 = add_constant_source(b.ravel(), 3, 4, 0.0)
    same_obj = b0 is b.ravel() or np.shares_memory(b0, b)
    e0 = float(np.abs(b0.reshape(3, 4) - b).max())
    ok &= same_obj and e0 == 0.0
    if verbose:
        print('  [4] H=0 no-op: shares_memory=%s  max|db|=%.1e' % (same_obj, e0))
    # [5] the source lands in exactly one coefficient and integrates to H
    src = source_coeffs(3, 4, H)
    nz = int(np.count_nonzero(src))
    ok &= (nz == 1) and (src[0, 0] == H) and (source_integral(H) == H)
    if verbose:
        print('  [5] source: %d non-zero coefficient (k=0,T_0), value %r, '
              'integral %r' % (nz, src[0, 0], source_integral(H)))
    # [6] H=0: the analytic state is theta == 0 and the reference is the frozen default
    ok &= (float(np.abs(conductive_theta(z, 0.0)).max()) == 0.0
           and float(np.abs(conductive_T(z, 0.0) - (1.0 - z)).max()) == 0.0
           and nu_bot_conductive(0.0) == 1.0 and nu_top_conductive(0.0) == 1.0
           and mean_T_conductive(0.0) == 0.5)
    if verbose:
        print('  [6] H=0 reference is the frozen background: '
              'theta_cond == 0, nu_bot == nu_top == 1, <T> == 0.5')
    if verbose:
        print('  heating self_test %s' % ('PASSED' if ok else 'FAILED'))
    return bool(ok)


if __name__ == '__main__':
    print('internal heating self test')
    self_test()
