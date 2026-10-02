"""specmc.stokes: variable-viscosity Stokes operators.

Home of the three layers of the Stokes solve:

    vv_stokes       per-wavenumber direct solve for eta(z) (``VVStokes`` /
                    ``VVStokesReduced``), plus the Chebyshev multiplication
                    matrix shared with the coupled operator
    stokes_coupled  the eta(x,z) coupled operator and its block preconditioners
                    (``CoupledStokes``, ``BlockSchurPrecond``)
    lagged_stokes   one lagged factorisation plus the adaptive Picard defect
                    correction (``LaggedStokes``, ``FreeSchurOperator``)

Nothing here knows about a physical scenario: the operators take an eta grid
and the model geometry.  Scenario assembly lives in ``specmc.models``.
"""
