"""Explicit bounds on the viscosity field ``eta(x, z)``.

Why this module exists
----------------------
The frozen reference rheology is Blankenbach et al. (1989) case 2a,

    eta(T) = exp( -ln(Delta eta_T) * T ),      T in [0, 1],

a law that is *calibrated on the temperature interval* ``T in [0, 1]``: over
that interval it spans exactly the nominal contrast ``Delta eta_T``
(``eta(T=1) = 1/Delta eta_T``, ``eta(T=0) = 1``).  Nothing in the law knows about
internal heating, so as soon as an internal source pushes ``T`` outside
``[0, 1]`` the exponential extrapolates: the viscosity keeps dropping without
bound and the viscosity *contrast* of the model is no longer the 1000 the whole
benchmark and the frozen solver stack were tuned for.  A measurement on the
32x48 production state gave the consequence: ``eta(T) + H`` leaves the
robustness envelope of the frozen theta solver for H >= 2, and even H = 1
diverges at t = 0.17.

Every mature mantle-convection code therefore bounds ``eta``.  What this module
adds is exactly that bound, applied pointwise *before* the field is truncated
back to the resolved modes (the anti-aliasing rule).

Provenance of the FORM (a plain additive clip on ``eta``)
---------------------------------------------------------
* ASPECT, ``Minimum viscosity`` / ``Maximum viscosity`` (Material model
  parameters): a "lower/upper cutoff for effective viscosity", applied as a
  plain clamp of the pointwise viscosity, ``std::clamp`` in
  ``visco_plastic.cc`` and ``rheology/diffusion_dislocation.cc``.  The manual's
  parameter table gives ``Minimum viscosity = 1e17`` Pa s and an upper bound
  many orders above the reference viscosity (``1e28`` Pa s in the current
  ``visco_plastic`` table; the value is material-model dependent), i.e. a
  *symmetric-in-magnitude* window around the model's reference viscosity.  The
  manual's section "Regularizing models with large coefficient
  variation" is a short pointer to the cell-averaging cookbook, not a smoothing
  recipe: ASPECT keeps the eta cutoff hard and regularizes other transitions
  instead (a C-infinity harmonic blend for ``Yield mechanism = limiter``, a
  harmonic diffusion/dislocation average).
* CitcomS, ``Viscosity_structures.c``: ``eta = max(eta_min, min(eta_max, eta))``
  after the rheology law (the ``vis_min`` / ``vis_max`` cutoffs); the manual
  presents the cutoff as the standard cure for an unconverged Stokes solve at
  large contrast.
* Underworld2, ``fn.maximum(fn.minimum(eta, eta_max), eta_min)``, the same
  clamp, used in the official Blankenbach and visco-plastic notebooks.
* G-ADOPT: the ``BoussinesqApproximation`` itself is unbounded (an exponential
  law), and the examples bound the viscosity field explicitly for the same
  reason (an explicit ``minimum``/``maximum`` on the viscosity function).

The literature on the *size* of the bound
-----------------------------------------
* Guerrero, Lowman & Tackley (2019), "Spurious transitions in convective
  regime due to viscosity clipping: ramifications for modeling planetary
  secular cooling", Geochem. Geophys. Geosyst. 20(7), 3450-3468,
  doi:10.1029/2019GC008385.  This shows that a viscosity clip *is* a first-order
  modelling decision: a fixed clip whose contrast falls below the
  stagnant-lid transition produces a *spurious regime change*.  Their remedy is
  a *dynamic* clip tied to the instantaneous CMB viscosity, not a smooth
  cutoff.  The practical rule: the bound follows the interval over which the
  law is calibrated and does not replace it.
* Blankenbach et al. (1989), Geophys. J. Int. 98, 23-38, Table 9 (case 2a):
  the calibration interval ``T in [0, 1]`` and ``Delta eta_T = 1000``.

The default this module implements
----------------------------------
    eta_min = 1 / Delta eta_T = 1e-3        eta_max = Delta eta_T = 1e3

i.e. the viscosity may not differ from the reference viscosity ``eta_ref = 1``
(which is what defines Ra) by more than the nominal contrast ``Delta eta_T``, in
either direction.  The frozen case-2a law spans ``[1e-3, 1]``, which is
strictly inside that window, so for the frozen reference configuration the bound is
*inactive* and the operation is bit-for-bit the old code path.  For ``H > 0``
the bound activates exactly when the heating pushes ``T`` out of the
calibration interval.

  * ``clip_eta``: the hard clip with the relative
    dead-band of :data:`CLIP_RTOL`.  Piecewise C^0.
  * ``smooth_clip_eta``: a C^2 compact-support replacement (the
    quintic Hermite smoothstep applied to ``ln eta``) that reduces to the hard
    clip in the zero-width limit.  See the smooth-clip section below.

The smooth (C^2) replacement
----------------------------
What the mature codes and the literature offer (every source claim
was checked against the code):

* ASPECT has no smoothing of the viscosity cutoff: ``visco_plastic.cc`` and
  ``rheology/diffusion_dislocation.cc`` use a plain ``std::clamp``, and the
  manual's "Regularizing models with large coefficient variation" section is a
  short pointer to the cell-averaging cookbook.  (There is no ``SmoothStep``
  utility in the ASPECT tree in the versions surveyed, master and the
  v2.3-v2.5 tags; that name is a *mesh-metric* taper in Underworld3.)  ASPECT's
  smooth forms live elsewhere: a C-infinity harmonic blend for
  ``Yield mechanism = limiter``, and a harmonic diffusion/dislocation average.
* CitcomS also clamps hard (``Viscosity_structures.c``, repeated); its optional
  ``use_ne_visc_smooth`` is a *nodal spatial average* (off by default), i.e.
  exactly the new grid/aliasing requirement this project avoids.
* Underworld2 exposes hard ``fn.minimum`` / ``fn.maximum``; UWGeodynamics wraps
  them in a limiter.  Neither has a smooth viscosity helper.
* G-ADOPT's ``BoussinesqApproximation`` does not limit ``mu``, but one of its
  own tests (``tests/adjoint_2d_cylindrical/forward.py``) implements a smooth
  maximum ``0.5*(x + a + sqrt((x-a)^2 + delta^2))`` with
  ``delta = mu_min_smoothing * mu_min``, commented with the reason a hard
  ``conditional`` "has a kink whose second derivative UFL drops".  That form is
  C-infinity but only *asymptotically* bounded (overshoot ``<= delta/2``).
* Underworld3's default ``yield_mode = "softmin"`` is a softplus smooth minimum
  with a documented ``yield_softness``; its retired ``"smooth"`` mode
  (``eta (1+f)/(1+f+f^2)``) is a documented failure ("it under-clipped the yield
  surface by ~50 %"), so it is not a model to copy.
* The method references are therefore the numerical-optimisation ones: Nesterov
  (2005) for smooth approximations of non-smooth functions, Boyd & Vandenberghe
  (Convex Optimization, section 3.1.5) for the log-sum-exp smooth maximum,
  Quilez's polynomial smooth-min, and the Hermite smoothstep (``3t^2-2t^3``
  C^1, ``6t^5-15t^4+10t^3`` C^2).
* The physical caveat is Guerrero, Lowman & Tackley (2019), above: a bound is a
  first-order modelling decision, and their remedy is a *dynamic* clip, not a
  smooth one.

Choice: the compact-support smoothstep clamp in ``ln eta``
(:func:`smooth_clip_eta`), because it is the only surveyed form that has all
four properties this solver needs at once:

1. it never leaves ``[eta_min, eta_max]`` (zero overshoot), the sqrt/softplus
   family would make the bound asymptotic and force a disclaimer;
2. it is C^2, which removes the kink the G-ADOPT comment complains about: the
   hard clamp's Fourier tail decays like ``k^-2``, the C^2 clamp's at least
   ``k^-5``, so the existing ``ETA_DEALIAS = 2`` stays sufficient and no new
   anti-aliasing requirement appears;
3. it is pointwise, so it composes with the padded-space evaluation exactly as
   the hard clip did (CitcomS's nodal average cannot);
4. ``width = 0.0`` is the hard clip *by construction*, the branch is
   taken before any log/exp, which is what makes the immune region bit for
   bit.

Documented cost: inside a transition band the local slope can exceed 1 by up to
~1.44x (the ramp is slightly steeper than the identity), the realised contrast
can be up to ``width`` wider in ``ln eta`` than the hard clip's (still inside
the window), and any ``width > 0`` perturbs the frozen case-2a field near
``T = 1``.  That is exactly why the default is ``0.0`` and the smooth path is
selected only by an explicit ``eta_smooth``.

The widest window the frozen law can reach
------------------------------------------
Earlier versions ran the internally heated ``eta(T)`` cases with ``eta_min = 1e-2``, a
1e4 window.  That is *not* the widest window this solver can use, and it is not
a property of the law: it is an extra bound that changes the rheology wherever
``T > log(100)/log(deta_T) = 0.667``.

The frozen temperature route already clips ``T`` into ``ETA_T_CLIP = (0, 2)``
*before* the exponential (`coupled_model._eta_from_T`), so the law
``eta = exp(-ln(deta_T) T)`` can never produce a viscosity below
``deta_T ** -ETA_T_CLIP[1]``, for ``deta_T = 1000`` that floor is exactly
``1e-6``.  :func:`reachable_window` returns that floor and its logarithmic
mirror, i.e. ``(deta_T**-2, deta_T**2)``.

Two properties make this the natural widest window:

* it is inert: with ``eta_min = deta_T**-2`` the eta clamp is a bit-for-bit
  no-op on *every* field the frozen solver can produce (the T-clip binds first,
  and the 1e-12 dead band absorbs the ulp-level violations at the floor), so it
  removes the physics-changing extra bound used earlier *by construction*
  instead of introducing a new hand-picked number;
* it is log-symmetric about the reference viscosity ``eta_ref = 1``
  (``eta_min * eta_max = 1``), the window shape used by the mature codes:
  ASPECT's ``Minimum viscosity`` / ``Maximum viscosity`` material-model cutoffs
  and CitcomS's ``VMIN`` / ``VMAX``.  The provenance of the *shape* is therefore
  the same as the hard clip; the provenance of the *number* is the solver's own
  ``ETA_T_CLIP``, not a guess.

Going wider (``eta_min < deta_T**-2``) changes nothing at all, the T-clip has
already saturated, and anything narrower is exactly the earlier, narrower bound.
"""
import numpy as np

__all__ = [
    "DEFAULT_MIN", "DEFAULT_MAX", "CLIP_RTOL", "SMOOTH_DEFAULT",
    "bounds_default", "reachable_window", "reachable_window_arrhenius",
    "reachable_window_pv", "effective_bounds", "validate_bounds",
    "clip_eta", "n_clipped", "validate_smooth", "smoothstep5",
    "n_in_transition", "smooth_clip_eta", "self_test",
]

#: ``eta_min`` / ``eta_max`` default (see the module docstring).  These are the
#: values ``ConvectionConfig`` reserves; an internal check compares the
#: config default and this module.
DEFAULT_MIN = 1e-3
DEFAULT_MAX = 1e3


#: Relative dead-band of the bound.  The law itself is evaluated in floating
#: point, so at the edge of its calibration interval (``T = 1`` exactly, i.e.
#: ``eta = 1/Delta eta_T``) the computed ``eta`` can land a few ulp *below* the
#: bound, measured on the frozen reference checkpoints: 393 of 184320 padded points
#: at ``eta = 0.0009999999999999985``, a relative violation of 1.5e-15.  A plain
#: clamp would then move those points and destroy the bit-for-bit immune region
#: for a reason that has nothing to do with the physics.  The bound is therefore
#: enforced only outside ``[eta_min (1-rtol), eta_max (1+rtol)]``: 12 orders of
#: magnitude above the roundoff and 10 orders below the smallest physically
#: meaningful violation (T = 1.01 already moves eta by 0.7 %).
CLIP_RTOL = 1e-12


def bounds_default():
    """-> ``(eta_min, eta_max)``, the documented default window."""
    return float(DEFAULT_MIN), float(DEFAULT_MAX)


def reachable_window(deta_T, t_clip=None):
    """-> the widest window the frozen ``eta(T)`` law can reach.

    ``t_clip`` defaults to the frozen temperature clip ``ETA_T_CLIP`` of
    `specmc._numerics`; the returned window is
    ``(deta_T ** -t_clip[1], deta_T ** t_clip[1])`` (see the widest-window section of
    the module docstring).  The lower edge is the smallest viscosity the
    T-clipped exponential law can produce, so applying it is a bit-for-bit no-op
    on every field the frozen solver can generate; the upper edge is its
    logarithmic mirror, which keeps the window log-symmetric about
    ``eta_ref = 1``, the window shape of ASPECT's ``Minimum viscosity`` /
    ``Maximum viscosity`` and CitcomS's ``VMIN`` / ``VMAX``.

    Raises ``ValueError`` for a non-positive/non-finite ``deta_T`` or a negative
    clip bound: a silently wrong window would be worse than no window (a bad
    window must not be silent).
    """
    if t_clip is None:
        from .._numerics import ETA_T_CLIP as _clip           # local import
        t_clip = _clip
    d = float(deta_T)
    t_hi = float(t_clip[1])
    if not (np.isfinite(d) and d > 0.0):
        raise ValueError("deta_T must be positive and finite (got %r)"
                         % (deta_T,))
    if not (np.isfinite(t_hi) and t_hi >= 0.0):
        raise ValueError("the temperature clip bound must be >= 0 (got %r)"
                         % (t_clip,))
    return d ** (-t_hi), d ** t_hi


def effective_bounds(eta_min, eta_max):
    """-> the window the dead-band enforces: ``(lo_eff, hi_eff)``."""
    return (float(eta_min) * (1.0 - CLIP_RTOL),
            float(eta_max) * (1.0 + CLIP_RTOL))


def reachable_window_arrhenius(deta_T, Ts, t_clip=None):
    """-> the widest window the T-only Arrhenius law can reach.

    Same construction as :func:`reachable_window`, with the law replaced.  The
    frozen route clips ``T`` into ``ETA_T_CLIP = (0, 2)`` and then evaluates
    ``eta(T)``; the Arrhenius law is strictly decreasing in ``T`` with
    ``eta(0) = 1`` exactly, so its range over the SAME clip is
    ``[eta(2), 1]``.

    Two consequences the calling code must know (they are why this is a
    SEPARATE function rather than a branch inside :func:`reachable_window`):

    * the floor is ``eta(2) = deta_T ** (-2 (1+Ts)/(2+Ts))``, NOT the
      exponential ``deta_T**-2``.  Since ``2(1+Ts)/(2+Ts) < 2`` for every
      finite ``Ts > 0``, the Arrhenius floor is HIGHER (the range over
      ``T in [0,2]`` is NARROWER) than the exponential one: at ``deta_T = 1000``
      the exponential window is ``(1e-6, 1e6)`` while the Arrhenius window is
      ``(1e-4, 1e4)`` at ``Ts = 1`` and ``(8.45e-4, 1.18e3)`` at ``Ts = 0.05``.
      This is the opposite of what a "steeper law needs a wider window" reading
      would suggest, and it is the correct one: the field's extreme points
      ``T = 0`` and ``T = 2`` are *shared* by both laws (``eta(0) = 1``), and
      the Arrhenius profile sits BELOW the exponential one in between, so it
      gets closer to neither extreme.  so the exponential window is a strict
      superset, so applying it to an Arrhenius run is also a no-op; the
      Arrhenius window is the tighter inert choice;
    * the window is still log-symmetric by construction
      (``(eta(2), 1/eta(2))``), keeping the mature codes' window SHAPE.

    ``t_clip`` defaults to the frozen temperature clip of ``specmc._numerics``.
    """
    from .arrhenius import eta_arrhenius_T                    # local import
    if t_clip is None:
        from .._numerics import ETA_T_CLIP as _clip
        t_clip = _clip
    d = float(deta_T)
    s = float(Ts)
    t_hi = float(t_clip[1])
    if not (np.isfinite(d) and d > 0.0):
        raise ValueError("deta_T must be positive and finite (got %r)"
                         % (deta_T,))
    if not (np.isfinite(s) and s > 0.0):
        raise ValueError("arrhenius_Ts must be positive and finite (got %r)"
                         % (Ts,))
    if not (np.isfinite(t_hi) and t_hi >= 0.0):
        raise ValueError("the temperature clip bound must be >= 0 (got %r)"
                         % (t_clip,))
    lo = float(eta_arrhenius_T(t_hi, d, s))
    return lo, 1.0 / lo


def reachable_window_pv(E, V, Ts, t_clip=None):
    """-> the widest window the pV(L) law can reach.

    The law is ``ln eta = -E T/(Ts (T+Ts)) + V p0_norm/(T+Ts)`` with

    * ``T`` clipped into the frozen ``ETA_T_CLIP`` (so ``T in [0, t_hi]``), and
    * ``p0_norm = 1 - z in [0, 1]``.

    The padded evaluation can pair the two bounds independently (the guard must
    bound every pair it is handed, not only the conduction line
    ``T = 1 - z``), so ``ln eta`` is minimised at ``(T = t_hi, p0_norm = 0)``
    and maximised at ``(T = 0, p0_norm = 1)``, it is decreasing in ``T`` and
    increasing in ``p0_norm`` for every ``Ts > 0``.  Hence

        lo = exp( -E t_hi / (Ts (t_hi + Ts)) )        (the T-only floor)
        hi = exp( +V / Ts )

    Not log-symmetric, unlike the exponential and T-only Arrhenius windows:
    pV makes the deep/cold end stiffer, so the upper edge is genuinely above 1
    whenever ``V > 0``.  With ``V = 0`` it reduces exactly to
    ``reachable_window_arrhenius`` (whose log-symmetric partner is ``1/lo``,
    larger than ``hi = 1``).  Applying the window is a no-op on every field the
    law can evaluate.
    """
    from .arrhenius import arrhenius_exponent_V               # local import
    if t_clip is None:
        from .._numerics import ETA_T_CLIP as _clip
        t_clip = _clip
    E = float(E)
    V = float(V)
    Ts = float(Ts)
    t_hi = float(t_clip[1])
    if not np.isfinite(E):
        raise ValueError("E must be finite (got %r)" % (E,))
    if not np.isfinite(V):
        raise ValueError("V must be finite (got %r)" % (V,))
    if not (np.isfinite(Ts) and Ts > 0.0):
        raise ValueError("Ts must be positive and finite (got %r)" % (Ts,))
    if not (np.isfinite(t_hi) and t_hi >= 0.0):
        raise ValueError("the temperature clip bound must be >= 0 (got %r)"
                         % (t_clip,))
    g_lo = float(arrhenius_exponent_V(t_hi, 0.0, E, V, Ts, p0_norm=0.0))
    g_hi = float(arrhenius_exponent_V(0.0, 0.0, E, V, Ts, p0_norm=1.0))
    if not (np.isfinite(g_lo) and np.isfinite(g_hi)):
        raise ValueError("the pV window is not finite (g_lo=%r g_hi=%r)"
                         % (g_lo, g_hi))
    return float(np.exp(g_lo)), float(np.exp(g_hi))


def validate_bounds(eta_min, eta_max):
    """Reject a bound pair that cannot be used, return the floats.

    Raises ``ValueError`` for a non-finite, non-positive or inverted pair.  The
    check is explicit: a silently ignored bound would be worse than
    no bound at all (a bound that does nothing is an error).
    """
    lo, hi = float(eta_min), float(eta_max)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError("eta bounds must be finite (got %r, %r)" % (lo, hi))
    if lo <= 0.0:
        raise ValueError("eta_min must be positive (got %r)" % (lo,))
    if hi <= lo:
        raise ValueError("eta_max must exceed eta_min (got %r, %r)" % (lo, hi))
    return lo, hi


def n_clipped(eta, eta_min, eta_max):
    """-> number of entries of ``eta`` outside the effective window."""
    lo, hi = effective_bounds(eta_min, eta_max)
    eta = np.asarray(eta)
    return int(np.count_nonzero((eta < lo) | (eta > hi)))


def clip_eta(eta, eta_min, eta_max):
    """Clamp ``eta`` into ``[eta_min, eta_max]`` (with the 1e-12 dead-band).

    Returns ``(eta_out, n_moved)``.  ``eta_out`` is the INPUT OBJECT unchanged
    when no entry is outside the effective window, so the immune region is
    bit-for-bit: no new array, no arithmetic, no rounding.  ``n_moved`` is the
    counter that every driver records as ``n_clip_eta``.
    """
    eta = np.asarray(eta, dtype=float)
    n = n_clipped(eta, eta_min, eta_max)
    if n == 0:
        return eta, 0
    return np.clip(eta, eta_min, eta_max), n


#: Default smoothing half-width, in units of ``ln eta``.  ``0.0`` selects the
#: hard clip of :func:`clip_eta`, the hard-clip code path, bit for bit (see the
#: smooth-clip section of the module docstring for the provenance).
SMOOTH_DEFAULT = 0.0


def validate_smooth(width, eta_min, eta_max):
    """Reject a smoothing width that cannot be used; return the float.

    ``width`` is a half-width in ``ln eta``.  It must be finite and >= 0, and
    ``2 * width`` must not exceed the log-width of the window: otherwise the two
    transition bands would meet, the map would lose its identity band and with
    it the "the interior is untouched" property that keeps the smooth clamp
    cheap and predictable.
    """
    w = float(width)
    if not np.isfinite(w):
        raise ValueError("eta_smooth must be finite (got %r)" % (width,))
    if w < 0.0:
        raise ValueError("eta_smooth must be >= 0 (got %r)" % (w,))
    span = float(np.log(float(eta_max)) - np.log(float(eta_min)))
    if w > 0.5 * span:
        raise ValueError(
            "eta_smooth=%r is too wide for the window [%r, %r]: the two "
            "transition bands would overlap (need eta_smooth <= %r)"
            % (w, eta_min, eta_max, 0.5 * span))
    return w


def smoothstep5(t):
    """Quintic Hermite smoothstep: 0 below 0, 1 above 1, C^2 at both ends.

    ``H(t) = 6 t^5 - 15 t^4 + 10 t^3`` on ``[0, 1]``, with
    ``H(0)=H'(0)=H''(0)=0`` and ``H(1)=1``, ``H'(1)=H''(1)=0``.
    """
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def _transition_mask(y, y_lo, y_hi, w):
    """Entries of ``y = ln eta`` strictly inside one of the two bands."""
    return (((y > y_lo) & (y < y_lo + w))
            | ((y > y_hi - w) & (y < y_hi)))


def n_in_transition(eta, eta_min, eta_max, width):
    """-> number of entries strictly inside one of the transition bands."""
    w = float(width)
    if w <= 0.0:
        return 0
    y = np.log(np.asarray(eta, dtype=float))
    return int(np.count_nonzero(
        _transition_mask(y, np.log(float(eta_min)), np.log(float(eta_max)), w)))


def smooth_clip_eta(eta, eta_min, eta_max, width):
    """C^2 compact-support clamp of ``eta``; ``width <= 0`` is the hard clip.

    The map is built in ``y = ln eta``, the natural coordinate of an
    exponential viscosity law, and the coordinate in which the window
    ``[1/Delta eta_T, Delta eta_T]`` is symmetric::

        Y       = y_lo + (y - y_lo) * H((y - y_lo)/w)      # lower bound
        y_out   = y_hi - (y_hi - Y) * H((y_hi - Y)/w)      # upper bound
        eta_out = exp(y_out)

    with ``H = smoothstep5``.  Properties (all checked in `self_test`):

    * exactly bounded: ``eta_out in [eta_min, eta_max]`` for every input,
      with no overshoot, unlike the softplus / sqrt family, which is only
      asymptotically bounded;
    * identity band: ``eta_out = eta`` (to roundoff) for
      ``y in [y_lo + w, y_hi - w]``;
    * C^2 in eta: ``H`` has two vanishing derivatives at both ends, and the
      composition of C^2 monotone maps is C^2; this is what removes the kink
      (and its ``k^-2`` Fourier tail) of the hard clamp;
    * monotone: ``smooth_clip_eta(exp(-ln(Delta eta_T) T))`` is strictly
      decreasing in ``T``, like the unbounded law and the hard clip, so the
      theta solver sees no new extrema;
    * ``width = 0`` dispatches to :func:`clip_eta` BEFORE any log/exp, so the
      hard-clip code path, the ``np.clip``, the 1e-12 dead-band and the "return
      the input object" no-op, survives bit for bit.

    Returns ``(eta_out, n_in_band)``.  ``n_in_band`` is the number of entries
    the smoothing reshaped (strictly inside a transition band); the model
    records it as ``n_smooth_eta``.  The hard-clip counter is still produced by
    :func:`n_clipped` and recorded as ``n_clip_eta``, so the hard clip and the smooth clip
    stay directly comparable.
    """
    w = float(width)
    if w <= 0.0:
        return clip_eta(eta, eta_min, eta_max)
    y_lo = float(np.log(float(eta_min)))
    y_hi = float(np.log(float(eta_max)))
    y = np.log(np.asarray(eta, dtype=float))
    n = int(np.count_nonzero(_transition_mask(y, y_lo, y_hi, w)))
    Y = y_lo + (y - y_lo) * smoothstep5((y - y_lo) / w)
    y_out = y_hi - (y_hi - Y) * smoothstep5((y_hi - Y) / w)
    return np.exp(y_out), n


# ------------------------------------------------------------------ self test
def self_test(verbose=True):
    """Structural checks of the bound (no solver needed).

    [1] the default window is the documented one and the frozen law's own range
        ``[1/deta_T, 1]`` is strictly inside it (the immune-region argument);
    [2] ``clip_eta`` returns the INPUT OBJECT, bit for bit, when nothing is
        outside the window, the immune-region guarantee;
    [3] outside the window it is exactly ``np.clip`` and ``n_moved`` is the
        number of moved entries;
    [4] ``validate_bounds`` rejects every unusable pair.
    """
    ok = True
    lo, hi = bounds_default()
    ok &= (lo, hi) == (1e-3, 1e3)

    # [1] the calibrated interval of the case-2a law is inside the window
    T = np.linspace(0.0, 1.0, 257)
    eta_cal = np.exp(-np.log(1000.0) * T)
    inside = bool(np.all((eta_cal >= lo) & (eta_cal <= hi)))
    ok &= inside
    if verbose:
        print('  [1] default window [%g, %g]; the case-2a law spans [%.6g, %.6g]'
              ' over T in [0,1] -> inside %s'
              % (lo, hi, eta_cal.min(), eta_cal.max(), inside))

    # [2] no-op: the same object comes back
    a = np.array([1.0, 0.5, 1e-3, 1.0000000000000002e-3])
    b, n = clip_eta(a, lo, hi)
    same_obj = b is a
    ok &= same_obj and (n == 0)
    if verbose:
        print('  [2] in-window array returned unchanged (same object %s, '
              'n_moved %d)' % (same_obj, n))

    # [3] real clip
    c = np.array([1e-9, 1e-3, 1.0, 1e9])
    d, n = clip_eta(c, lo, hi)
    exact = bool(np.array_equal(d, np.array([lo, 1e-3, 1.0, hi])))
    ok &= exact and (n == 2)
    if verbose:
        print('  [3] out-of-window entries clipped exactly (%s), n_moved %d '
              '(expected 2)' % (exact, n))

    # [4] unusable windows
    bad = 0
    for pair in ((0.0, 1e3), (-1.0, 1e3), (1e2, 1e-2), (1e-3, np.inf),
                 (1e-3, 1e-3)):
        try:
            validate_bounds(*pair)
        except ValueError:
            bad += 1
    ok &= (bad == 5)
    if verbose:
        print('  [4] validate_bounds rejected %d/5 unusable windows' % bad)

    # [5] the dead-band: a field 1e-15 relative below the bound is NOT moved
    #     (roundoff), one 1e-3 relative below IS
    roud = np.array([lo * (1.0 - 1e-15), hi * (1.0 + 1e-15)])
    _, n5 = clip_eta(roud, lo, hi)
    real = np.array([lo * (1.0 - 1e-3)])
    _, n6 = clip_eta(real, lo, hi)
    ok &= (n5 == 0) and (n6 == 1)
    if verbose:
        print('  [5] dead-band: 1e-15-relative violation not clipped (%d), '
              '1e-3-relative violation clipped (%d)' % (n5, n6))

    # ---------------------------------------------------- the smooth clip: the smooth
    # [6] width = 0 is the hard-clip path, bit for bit (object identity included)
    a = np.array([1e-9, 1e-3, 0.5, 1.0, 1e9])
    s0, n0 = smooth_clip_eta(a, lo, hi, 0.0)
    h0, m0 = clip_eta(a, lo, hi)
    same0 = bool(np.array_equal(s0, h0)) and (n0 == m0)
    inwin = np.array([1.0, 0.5, 1e-3])
    s_in, _ = smooth_clip_eta(inwin, lo, hi, 0.0)
    ok &= same0 and (s_in is inwin)
    if verbose:
        print('  [6] width=0 == clip_eta bit for bit (%s); in-window input '
              'returned as the same object (%s)' % (same0, s_in is inwin))

    # [7] width > 0: exactly bounded, identity band, monotone, and C^2 where
    #     the hard clip is only C^0 (a derivative jump of 1 at each joint)
    w = 1.0
    yy = np.linspace(np.log(lo) - 3.0, np.log(hi) + 3.0, 4001)
    e_in = np.exp(yy)
    e_sm, n7 = smooth_clip_eta(e_in, lo, hi, w)
    bounded = bool(np.all((e_sm >= lo * (1.0 - 1e-12))
                          & (e_sm <= hi * (1.0 + 1e-12))))
    band = (yy >= np.log(lo) + w) & (yy <= np.log(hi) - w)
    id_err = float(np.abs(np.log(e_sm[band]) - yy[band]).max())
    # yy increases, so a monotone clamp must be non-decreasing in yy (and is
    # therefore strictly decreasing in T for the exponential law)
    mono = bool(np.all(np.diff(e_sm) >= -1e-9 * hi))
    d_sm = np.gradient(np.log(e_sm), yy)
    d_hd = np.gradient(np.log(np.clip(e_in, lo, hi)), yy)
    jump_sm = float(np.abs(np.diff(d_sm)).max())
    jump_hd = float(np.abs(np.diff(d_hd)).max())
    smooth_c1 = (jump_sm < 0.05) and (jump_hd > 0.4)
    ok &= bounded and (id_err < 1e-13) and mono and smooth_c1
    if verbose:
        print('  [7] w=%g: bounded exactly %s; identity-band err %.1e; '
              'monotone %s; max|d(slope)| smooth %.1e vs hard %.1e -> C^1 %s'
              % (w, bounded, id_err, mono, jump_sm, jump_hd, smooth_c1))

    # [8] the band count and the w -> 0 limit
    _, nb = smooth_clip_eta(e_in, lo, hi, w)
    cnt_ok = (nb == n_in_transition(e_in, lo, hi, w)) and nb > 0
    e_tiny, _ = smooth_clip_eta(e_in, lo, hi, 1e-12)
    lim_err = float(np.abs(e_tiny - np.clip(e_in, lo, hi)).max()
                    / max(np.clip(e_in, lo, hi).max(), 1e-300))
    ok &= cnt_ok and (lim_err < 1e-14)
    if verbose:
        print('  [8] transition-band count %d (consistent %s); w=1e-12 vs hard '
              'clip rel %.1e' % (nb, cnt_ok, lim_err))

    # [9] unusable smoothing widths
    badw = 0
    for pair in ((-1.0, lo, hi), (np.nan, lo, hi), (np.inf, lo, hi),
                 (0.5 * np.log(hi / lo) + 1e-9, lo, hi)):
        try:
            validate_smooth(*pair)
        except ValueError:
            badw += 1
    ok &= (badw == 4)
    if verbose:
        print('  [9] validate_smooth rejected %d/4 unusable widths' % badw)

    # ------------------------------------------- the reachable window
    # [10] the window is DERIVED, not typed: its lower edge is the frozen
    #      T-clip's own eta floor, and it is log-symmetric about eta_ref = 1
    rlo, rhi = reachable_window(1000.0)
    floor = float(np.exp(-np.log(1000.0) * 2.0))
    exact_value = (rlo, rhi) == (1e-6, 1e6)
    logsym = (rlo * rhi == 1.0)
    # the T-clipped law's floating-point floor sits ONE ULP ABOVE deta_T**-2
    # (1.0000000000000004e-06 vs 1e-06), i.e. strictly inside the window: that
    # roundoff, not a choice, is why the window is inert.
    matches_floor = bool(abs(floor - rlo) <= 8.0 * np.finfo(float).eps * rlo)
    ok &= exact_value and logsym and matches_floor
    if verbose:
        print('  [10] reachable_window(1000) = (%g, %g) exact %s; '
              'log-symmetric %s; T-clip floor %.17g is inside to roundoff %s'
              % (rlo, rhi, exact_value, logsym, floor, matches_floor))

    # [11] ... and it is INERT: a field spanning T in [-1, 3] (so the T-clip
    #      saturates both ends) comes back as the SAME object, n_moved = 0
    T_test = np.linspace(-1.0, 3.0, 401)
    eta_test = np.exp(-np.log(1000.0) * np.clip(T_test, 0.0, 2.0))
    out11, n11 = clip_eta(eta_test, rlo, rhi)
    inert = (n11 == 0) and (out11 is eta_test)
    ok &= inert
    if verbose:
        print('  [11] reachable window is inert on T in [-1, 3]: n_moved %d, '
              'same object %s' % (n11, out11 is eta_test))

    # [12] unusable deta_T / clip bounds raise instead of returning nonsense
    bad12 = 0
    for bad in (0.0, -1.0, np.inf, np.nan):
        try:
            reachable_window(bad)
        except ValueError:
            bad12 += 1
    try:
        reachable_window(1000.0, (0.0, -1.0))
    except ValueError:
        bad12 += 1
    ok &= (bad12 == 5)
    if verbose:
        print('  [12] reachable_window rejected %d/5 unusable inputs' % bad12)

    # ------------------------------------- Arrhenius: the reachable window
    # [13] its floor is D**(-2(1+Ts)/(2+Ts)), HIGHER than the exponential
    #      D**(-2) for every finite Ts (the Arrhenius profile sits below the
    #      exponential one between the shared endpoints, so it approaches
    #      neither extreme).  Hence the Arrhenius window is a SUBSET of the
    #      exponential one, and both are inert on the fields their own law can
    #      produce.  It is log-symmetric, and at Ts -> inf it converges to the
    #      exponential window.
    from .arrhenius import eta_arrhenius_T as _eta_a
    D13 = 1000.0
    rlo_e, rhi_e = reachable_window(D13)
    for Ts in (1.0, 0.5, 0.2, 0.1, 0.05):
        alo, ahi = reachable_window_arrhenius(D13, Ts)
        exact = abs(alo - float(_eta_a(2.0, D13, Ts))) < 1e-15 * alo
        logsym = (alo * ahi == 1.0)
        subset = (alo >= rlo_e * (1.0 - 1e-15)) and (ahi <= rhi_e * (1.0 + 1e-15))
        pred = D13 ** (-2.0 * (1.0 + Ts) / (2.0 + Ts))
        pred_ok = abs(alo / pred - 1.0) < 1e-12
        ok &= bool(exact and logsym and subset and pred_ok)
        if verbose:
            print('  [13] Ts=%-5g window (%.6e, %.6e): exact %s, log-sym %s, '
                  'SUBSET of the exponential window (%.3e, %.3e) %s, '
                  '== D**(-2(1+Ts)/(2+Ts)) %s'
                  % (Ts, alo, ahi, exact, logsym, rlo_e, rhi_e, subset,
                     pred_ok))
    alo_inf, ahi_inf = reachable_window_arrhenius(D13, 1e9)
    lim_ok = abs(alo_inf / rlo_e - 1.0) < 1e-6
    ok &= lim_ok
    _bad13 = 0
    for args in ((1000.0, 0.0), (1000.0, -1.0), (0.0, 1.0), (-1.0, 1.0),
                 (np.inf, 1.0)):
        try:
            reachable_window_arrhenius(*args)
        except ValueError:
            _bad13 += 1
    try:
        reachable_window_arrhenius(1000.0, 1.0, (0.0, -1.0))
    except ValueError:
        _bad13 += 1
    ok &= (_bad13 == 6)
    if verbose:
        print('  [13] Ts=1e9 -> exponential window (rel %.1e); unusable inputs '
              'rejected %d/6' % (abs(alo_inf / rlo_e - 1.0), _bad13))

    # ------------------------------------------- the pV(L) reachable window
    # [14] V = 0 reproduces the T-only floor exactly; V > 0 lifts the CEILING to
    #      exp(V/Ts) (not log-symmetric); the window covers the law's range on
    #      the reference profile and unusable inputs raise.
    from .arrhenius import arrhenius_exponent_V as _expV
    from .reference_state import REFERENCE_DEFAULTS as _RC, calibrate_pv as _cal
    _rec = _cal(_RC, anchor="Ea")
    _lo0, _hi0 = reachable_window_pv(_rec["E"], 0.0, _rec["Ts"])
    _lo1, _hi1 = reachable_window_pv(_rec["E"], _rec["V"], _rec["Ts"])
    _floor_ok = abs(_lo1 / _lo0 - 1.0) < 1e-15
    _ceil_ok = abs(_hi1 - float(np.exp(_rec["V"] / _rec["Ts"]))) <= 1e-12 * _hi1
    _mono_ok = _hi1 > _hi0
    _T = np.linspace(0.0, 1.0, 2001)
    _g = np.asarray(_expV(_T, 1.0 - _T, _rec["E"], _rec["V"], _rec["Ts"]))
    _cover_ok = bool(float(np.exp(_g.min())) >= _lo1 * (1.0 - 1e-12)
                     and float(np.exp(_g.max())) <= _hi1 * (1.0 + 1e-12))
    _bad14 = 0
    for args in ((1.0, 0.0, 0.0), (1.0, 0.0, -1.0), (np.inf, 0.0, 1.0),
                 (1.0, np.inf, 1.0)):
        try:
            reachable_window_pv(*args)
        except ValueError:
            _bad14 += 1
    try:
        reachable_window_pv(1.0, 0.0, 1.0, (0.0, -1.0))
    except ValueError:
        _bad14 += 1
    ok &= bool(_floor_ok and _ceil_ok and _mono_ok and _cover_ok
               and (_bad14 == 5))
    if verbose:
        print('  [14] pV window: V=0 floor == T-only floor %s, ceiling '
              'exp(V/Ts) %s, ceiling lifted by V %s, covers the reference '
              'range %s; unusable inputs rejected %d/5'
              % (_floor_ok, _ceil_ok, _mono_ok, _cover_ok, _bad14))
        print('  eta_bounds self_test %s' % ('PASSED' if ok else 'FAILED'))
    return bool(ok)


if __name__ == '__main__':
    print('eta_bounds self test')
    self_test()

