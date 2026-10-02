"""Single source of truth for the CFL-safety policy.

Policy (after the operator and anti-aliasing corrections)
---------------------------------------
* global DEFAULT safety = 40          (was 5)
* HARD CAP            = 80           (was 10)
* no automatic dependence on deta_T  (one policy for every case)

Evidence (the production 64x96 campaign)
----------------------------------------
64x96, started from the t = 0.25 production checkpoint, FIXED dt = F x CFL(1),
exposure Dt = 0.02, failure on grid amplitude > 3x, Chebyshev tail > 10x, or
solver aborts.  With the corrected coupled operator (conjugate half of the
x-convolution) and the alias-free eta:

    F      20        40        80        112       160       224       320
    res    stable    stable    stable    stable    stable    stable    stable
    dNu*   0         +3.9e-6   +5.3e-6   +1.3e-5   +3.1e-5   +6.4e-5   +1.4e-4

    (*) final Nu minus Nu(F=20) on the same window; the F <= 80 plateau is
    flat to 5e-6 (5e-7 relative), and no stability ceiling was found up to
    F = 320 (dt = 6.8e-5, 320x the global CFL).

So the dt-CONVERGED range is F <= ~110 and F <= 80 is converged to <1e-6
relative.  DEFAULT = 40 sits in the middle of that plateau and CAP = 80 is its
edge, with a 4x stability margin to the smallest F at which the dt error
becomes measurable (320) and a 16x margin to the old, much weaker limit.

For comparison, an earlier campaign measured F* in [56, 64) with the SHIPPED
(pre-fix) operator, where F = 64, 80 and 160 all blew up inside the same window.

Why the old policy said 5/10
----------------------------
The measured 32x48 case-2a limit was dt ~ 5-10 x CFL, and dt = 20-26 x CFL was
not converged (13 % amplitude / 17 % Nu differences) and blew up around
t = 0.13.  That campaign ran with the pre-fix operator; the "2217 x CFL" ceiling
of the BDF2 self-test was an artefact of a 20-step / 20x-amplitude protocol
(measured in the earlier campaign).

Usage
-----
    from cfl_policy import CFL_SAFETY_DEFAULT, CFL_SAFETY_MAX, cfl_dt

    dt = cfl_dt(m, s, safety=CFL_SAFETY_DEFAULT, dtmax=4e-5)

`cfl_dt` clamps `safety` into [0, CFL_SAFETY_MAX] so a stale driver default
(e.g. the old 50) can never silently exceed the cap again.
"""
import numpy as np

__all__ = ["CFL_SAFETY_DEFAULT", "CFL_SAFETY_MAX", "clamp_safety", "cfl_dt"]

#: global default advective-CFL safety factor for every case
CFL_SAFETY_DEFAULT = 40.0
#: hard upper bound; dt is never allowed above this multiple of the global CFL
CFL_SAFETY_MAX = 80.0


def clamp_safety(safety, lo=0.0):
    """Clamp a requested safety factor into [lo, CFL_SAFETY_MAX]."""
    return float(np.clip(float(safety), float(lo), CFL_SAFETY_MAX))


def cfl_dt(model, that, safety=CFL_SAFETY_DEFAULT, dtmax=4.0e-5, dtmin=1e-10,
           clamp=True):
    """dt = min(dtmax, clamp(safety) * model.cfl_dt(that, safety=1)).

    Identical to the driver idiom `min(dtmax, m.cfl_dt(s, safety=FAC,
    dtmax=dtmax))` except that the safety factor is capped at
    `CFL_SAFETY_MAX`.
    """
    fac = clamp_safety(safety) if clamp else float(safety)
    return float(min(dtmax, model.cfl_dt(that, safety=fac, dtmax=dtmax,
                                         dtmin=dtmin)))