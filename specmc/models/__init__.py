"""specmc.models: the assembled physical models (scenario -> solver).

Home of the production model modules:

    convection_modes.py  ``make_convection`` (scenario factory) and the
                         eta(z) model ``RBCVariableViscosity``
    coupled_model.py     ``RBCCoupled``: eta(T(x,z,t)) with the lagged coupled
                         Stokes operator and the BDF2 implicit temperature step

and of the three explicit scenario recipes that ``specmc.api.model_spec``
dispatches to, one file per implemented ``viscosity_model``:

    const.py  ``viscosity_model='const'``  -> ``rbc_galerkin.RBC``  (isoviscous)
    eta_z.py  ``viscosity_model='eta_z'``  -> ``RBCImplicitTheta``  (eta(z))
    eta_T.py  ``viscosity_model='eta_T'``  -> ``RBCCoupled``        (reference default)

The two model modules *compose*: they pick a discretisation from
``specmc.core``, an eta law from ``specmc.physics`` and a Stokes operator from
``specmc.stokes``, and expose a life-cycle (``seed`` / ``step`` /
``diagnostics``).  The three recipe modules hold no physics at all: they are
the single place where a ``ConvectionConfig`` becomes constructor arguments,
and each of them is checked against the class it must build.

The public entry point is not here but in ``specmc.api``, which validates
the config and dispatches on ``viscosity_model``.
"""
