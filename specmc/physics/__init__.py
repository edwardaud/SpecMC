"""specmc.physics: the constitutive physics of the convection problem.

Home of the viscosity laws: ``viscosity.py`` is the single source of truth for
every eta formula the solver uses (eta(T), eta(z) on the conductive background,
and the law/parameter mapping).

``heating.py`` is the single source of truth for the internal heating
internal heating: the constant volumetric source H of
``d(theta)/dt + u.grad(theta) - u_z = lap(theta) + H``, its analytic
conductive reference state, and the energy-budget identity
``d<T>/dt = Nu_bot - Nu_top + H``.  It is implemented.

``dissipation.py`` is the single source of truth for the viscous
viscous dissipation: ``Phi = 2 eta (eps - (1/3) tr(eps) 1) : eps``
with the coefficient ``Di/Ra`` (``Di = dissipation_number``), the padded
evaluation pipeline that reuses ``eta_pad_factor``, the exact momentum-balance
identity ``<Phi> = Ra <theta u_z>`` and the ``H + Phi`` energy
budget.  It is implemented; the TALA equation layer is implemented
(``compressibility='tala'``, ``specmc.stokes.stokes_tala``).

``eta_bounds.py`` is the single source of truth for the explicit viscosity
viscosity bounds: ``clip_eta`` and the smooth-clip ``smooth_clip_eta``, the
provenance of the default window ``[1/Delta eta_T, Delta eta_T]``, and the
counter that every driver records as ``n_clip_eta``.  A bound is a *physics*
statement (over which interval is the rheology law calibrated?), which is why it
lives here and not in the solver.

``reference_state.py`` is the single source of truth for the
reference state: the reference hydrostatic pressure
``p0(z) = rho0 g d (1 - z)`` that the pV term multiplies, the eight dimensional
constants that anchor the dimensionless groups, the ONE pV calibration
procedure ``calibrate_pv``, and, reserved for a later TALA implementation, King's
reference profiles ``rho_bar/T_bar/p_bar_king``.  Nothing in the package may
re-type ``rho g z``.

``compressibility.py`` is the single source of truth for the TALA
reference state: the one ``Di`` (which is ``dissipation_number``,
shared with viscous dissipation), the Grueneisen ``gamma``, and the two derived objects the
equation layer consumes: the constant logarithmic density slope
``a = rho_bar'/rho_bar = -Di/gamma`` and King's profiles ``rho_bar/T_bar/p_bar``
(read from ``reference_state.py``, not re-implemented).  ``Di = 0`` is the exact
Boussinesq limit.  It is implemented.

A module in this subpackage owns *physics only*: it evaluates a material
property (or a source) from the state.  It does not know how that property is
discretised, dealiased or solved for; that is ``specmc.core`` and
``specmc.stokes``.
"""
