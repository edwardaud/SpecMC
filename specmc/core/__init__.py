"""specmc.core: physics-free numerical kernel of the solver.

Home of the modules that describe *how* the equations are discretised and
marched, independently of *which* physics is switched on:

    rbc_galerkin    Chebyshev-tau / Chebyshev-Galerkin discretisation and the
                    isoviscous reference solver (``RBC``)
    cfl_policy      single source of truth for the CFL-limited dt policy
    krylov_fgmres   matrix-free FGMRES used by the implicit temperature solve
    implicit_theta  BE / BDF2 implicit temperature step (``RBCImplicitTheta``)
    bench_ic        the three benchmark initial conditions

Viscosity laws (and, later, heating / compressibility) live in
``specmc.physics``; the Stokes operators in ``specmc.stokes``; the assembled
production models in ``specmc.models``.

This ``__init__`` imports nothing.  The submodules are imported by
``specmc.models`` and by the internal cross-check scripts (as
``python -m specmc.core.<module>`` for their self-tests), and an eager import
here would only widen the import graph of every entry point.
"""
