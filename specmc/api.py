"""`specmc` public API: config -> model -> march.

This module adds no physics.  Nothing here redefines a solver; it is a thin,
validated translation layer

    ConvectionConfig  --(validate / model_spec)-->  production model instance

plus two march helpers and a checkpoint reader so that an external driver can
drive the model without re-implementing the production loop.

The translation itself is not a chain of branches in this file any more: every
implemented ``viscosity_model`` has one explicit recipe module under
``specmc.models`` (``const``, ``eta_z``, ``eta_T``), and this file only
validates the config, dispatches to that module and checks the class that came
back.  See :mod:`specmc.models.const` for the convention every recipe module
follows (``NAME`` / ``CLS`` / ``BUILDER`` / ``kwargs`` / ``build`` / ``spec``).

Layout
-----------
Every production module lives inside the package and every import below is
intra-package.  The pre-migration flat module names are gone: the call sites were
migrated to ``specmc.*`` and the shims were deleted.  The layout is asserted by
the package self-tests: no flat shim, no flat import.
"""
from __future__ import annotations

import numpy as np

from . import _numerics as NUM                                   # noqa: E402
from .config import (ConvectionConfig, normalize_compressibility,
                     normalize_viscosity_model)                  # noqa: E402
from .core.cfl_policy import cfl_dt as _policy_dt                # noqa: E402
from .io import checkpoint as _ckpt                              # noqa: E402
from .models import const as _model_const                        # noqa: E402
from .models import eta_T as _model_eta_T                        # noqa: E402
from .models import eta_z as _model_eta_z                        # noqa: E402

__all__ = [
    "MODEL_MODULES", "MODEL_CLASSES", "numerics", "model_spec", "model_kwargs",
    "build_model", "describe", "production_dt", "load_checkpoint", "march",
    "march_adaptive",
]

#: implemented ``viscosity_model`` -> the explicit module that owns its recipe
#: (``specmc.models.{const,eta_z,eta_T}``).  This mapping is the whole dispatch:
#: there is no per-scenario branch left in this file.
#:
#: The Arrhenius law: ``'arrhenius'`` shares the ``eta_T`` recipe module on purpose.  The
#: two scenarios differ in exactly ONE constructor argument, ``arrhenius_Ts``,
#: which selects the law inside ``RBCCoupled._eta_from_T``, so routing both
#: through the same recipe is what makes the comparison between them a
#: single-variable experiment.
MODEL_MODULES = {
    "const": _model_const,
    "eta_z": _model_eta_z,
    "eta_T": _model_eta_T,
    "arrhenius": _model_eta_T,
}

#: canonical viscosity model -> the legacy class the production numerics stack
#: instantiates.  Checked by `build_model` on every build.
MODEL_CLASSES = {name: mod.CLS for name, mod in MODEL_MODULES.items()}


# --------------------------------------------------------------------------
# introspection
# --------------------------------------------------------------------------
def numerics():
    """-> the fixed internal numerical constants (a copy)."""
    return NUM.as_dict()


def model_spec(cfg: ConvectionConfig):
    """-> the exact construction recipe for ``cfg`` (class, builder, kwargs).

    This is the single place where a physical scenario is translated into
    solver arguments; ``build_model`` is a three-line consequence of it, and
    internal cross-check scripts can inspect it without building anything.

    The scenario-specific part of the recipe lives in the ``specmc.models``
    module selected by ``viscosity_model``; here the config is validated and the
    model is dispatched.
    """
    if not isinstance(cfg, ConvectionConfig):
        raise TypeError("expected ConvectionConfig, got %r" % (type(cfg),))
    cfg.validate()
    vm = normalize_viscosity_model(cfg.viscosity_model)
    comp = normalize_compressibility(cfg.compressibility)

    module = MODEL_MODULES.get(vm)
    if module is None:
        raise NotImplementedError("viscosity_model=%r is not implemented"
                                  % (vm,))
    return module.spec(cfg, comp)


def model_kwargs(cfg: ConvectionConfig):
    """-> the constructor keyword arguments used for ``cfg`` (a copy)."""
    return dict(model_spec(cfg)["kwargs"])


def build_model(cfg: ConvectionConfig):
    """Build the solver for ``cfg`` and check that it is the documented class."""
    spec = model_spec(cfg)
    model = MODEL_MODULES[spec["model"]].build(**spec["kwargs"])
    if type(model) is not spec["cls"]:
        raise AssertionError(
            "internal mapping error: viscosity_model=%r built %s, expected %s"
            % (spec["model"], type(model).__name__, spec["cls"].__name__))
    return model


def describe(cfg: ConvectionConfig):
    """-> a JSON-safe description of what ``cfg`` will run (config + numerics)."""
    spec = model_spec(cfg)
    return dict(config=cfg.canonical(), model=spec["model"],
                model_class=spec["cls"].__name__, builder=spec["builder"],
                ctor_kwargs={k: (None if v is None else v)
                             for k, v in spec["kwargs"].items()},
                numerics=numerics())


# --------------------------------------------------------------------------
# marching
# --------------------------------------------------------------------------
def production_dt(model, that):
    """-> dt of the production CFL policy: min(DTMAX, CFL_SAFETY * global CFL).

    ``model.cfl_dt`` already solves Stokes; the model memoises that solve
    (`RBCCoupled.stokes`), so a following ``step`` reuses it.
    """
    return float(_policy_dt(model, that, safety=NUM.CFL_SAFETY,
                            dtmax=NUM.DTMAX))


def load_checkpoint(path):
    """-> ``(theta_coefficients, meta)`` from a driver checkpoint ``.npz``.

    The drivers write a state with ``np.savez_compressed`` (rbccoupled: ``that``
    plus ``t``, ``n``, ``fac``); anything else in the file is ignored.  The
    implementation lives in :mod:`specmc.io.checkpoint`; this is the frozen
    public entry point.
    """
    return _ckpt.load(path)


def march(model, that, dt, nsteps, keep_states=False, on_step=None):
    """Fixed-dt march of ``nsteps`` steps (a frozen reference protocol).

    Returns ``(state, info)``; ``info['states']`` holds every intermediate state
    when ``keep_states`` is set.
    """
    dt = float(dt)
    nsteps = int(nsteps)
    s = np.array(that, copy=True)
    states = np.empty((nsteps,) + s.shape, dtype=complex) if keep_states else None
    for n in range(nsteps):
        s = model.step(s, dt)
        if states is not None:
            states[n] = s
        if on_step is not None:
            on_step(n + 1, (n + 1) * dt, s)
    info = dict(steps=nsteps, t=nsteps * dt, dt=dt)
    if states is not None:
        info["states"] = states
    return s, info


def march_adaptive(model, that, tstop, safety=None, dtmax=None, on_step=None):
    """CFL-limited march to ``tstop`` with the production policy by default.

    ``safety`` / ``dtmax`` exist for *frozen reference protocols* only: the
    isoviscous regression runs with the legacy ``safety=0.4, dtmax=5e-6`` and
    must reproduce its stored target.  They are not part of the public
    interface: a physical scenario does not choose its CFL policy.
    """
    safety = NUM.CFL_SAFETY if safety is None else float(safety)
    dtmax = NUM.DTMAX if dtmax is None else float(dtmax)
    s = np.array(that, copy=True)
    t = 0.0
    n = 0
    dt = float("nan")
    while t < float(tstop):
        dt = float(_policy_dt(model, s, safety=safety, dtmax=dtmax))
        s = model.step(s, dt)
        t += dt
        n += 1
        if on_step is not None:
            on_step(n, t, s)
    return s, dict(steps=n, t=t, dt=dt, safety=safety, dtmax=dtmax)
