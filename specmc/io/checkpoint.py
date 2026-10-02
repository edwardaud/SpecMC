"""Run-tree checkpoint I/O for `specmc`.

The frozen reference production driver writes a march state as a compressed ``.npz``
on a fixed time grid and resumes from it.  That write/read logic used to be
duplicated inside the driver; it lives here now so that the production API
(`specmc.api.load_checkpoint`), the driver and any future driver share one
implementation.

Contract (unchanged from the frozen reference driver)
----------------------------------------------
* a state file is ``<dir>/<prefix>_<t:%.6f>.npz`` holding at least the complex
  coefficient array ``that`` plus the scalars ``t``, ``n`` and ``fac``;
* the write is atomic: a sibling ``<name>.npz.tmp.npz`` is written first and then
  ``os.replace``-d to the final name.  The temporary file must carry the ``.npz``
  suffix itself, because ``numpy.savez_compressed`` appends ``.npz`` when the
  target has no such suffix (an early smoke test produced
  ``t_0.002005.npz.tmp.npz`` and then failed the rename);
* reading is tolerant: a missing ``t`` / ``n`` / ``fac`` becomes ``nan`` /
  ``None`` / ``None`` and the extra keys are reported in ``meta['keys']``.

Nothing here is physics and nothing here changes the frozen file layout.
"""
from __future__ import annotations

import os

import numpy as np

__all__ = ["DEFAULT_PREFIX", "filename", "path_for", "save", "list_",
           "latest", "load"]

#: state prefix written by the production march (``t_<time>.npz``)
DEFAULT_PREFIX = "t"

#: state prefix of the periodically KEPT (resume) states
KEEP_PREFIX = "keep"


def filename(prefix, t):
    """-> the checkpoint file NAME of ``prefix`` at time ``t``."""
    return "%s_%.6f.npz" % (prefix, float(t))


def path_for(directory, prefix, t):
    """-> the checkpoint file PATH of ``prefix`` at time ``t``."""
    return os.path.join(directory, filename(prefix, t))


def save(directory, that, t, n=None, fac=None, prefix=DEFAULT_PREFIX):
    """Atomically write one state to ``directory``; -> the final path."""
    os.makedirs(directory, exist_ok=True)
    fn = path_for(directory, prefix, t)
    tmp = fn + ".tmp.npz"
    np.savez_compressed(tmp, that=that, t=float(t), n=n, fac=fac)
    os.replace(tmp, fn)
    return fn


def list_(directory, prefix=DEFAULT_PREFIX):
    """-> ``[(t, path)]`` of the ``prefix`` states in ``directory``, by time."""
    out = []
    if not os.path.isdir(directory):
        return out
    head = prefix + "_"
    for f in os.listdir(directory):
        if f.startswith(head) and f.endswith(".npz"):
            try:
                out.append((float(f[len(head):-4]), os.path.join(directory, f)))
            except ValueError:
                pass
    return sorted(out)


def latest(directory, prefix=DEFAULT_PREFIX):
    """-> ``(t, path)`` of the newest state, or ``None`` if there is none."""
    states = list_(directory, prefix)
    return states[-1] if states else None


def load(path):
    """-> ``(theta_coefficients, meta)`` from one checkpoint ``.npz``.

    ``FileNotFoundError`` for a missing file and ``ValueError`` for a file
    without a ``that`` field, exactly the contract of
    :func:`specmc.api.load_checkpoint`, which delegates here.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    try:
        files = sorted(z.files)
        if "that" not in z:
            raise ValueError("%s has no 'that' field (keys: %s)"
                             % (path, files))
        that = np.asarray(z["that"], dtype=complex)
        meta = dict(path=os.path.abspath(path), keys=files,
                    t=(float(z["t"]) if "t" in files else float("nan")),
                    n=(int(z["n"]) if "n" in files else None),
                    fac=(float(z["fac"]) if "fac" in files else None))
    finally:
        if hasattr(z, "close"):
            z.close()
    return that, meta


# ------------------------------------------------------------------ self-test
def self_test():
    """Round-trip a state through a temporary directory; -> True."""
    import tempfile

    that = (np.arange(12, dtype=float).reshape(3, 4)
            + 1j * np.arange(12, dtype=float).reshape(3, 4))
    with tempfile.TemporaryDirectory() as d:
        for t in (0.01, 0.02, 0.005):
            save(d, that, t, n=7, fac=40.0)
        save(d, 2.0 * that, 0.05, n=9, fac=20.0, prefix=KEEP_PREFIX)
        periodic = list_(d)
        kept = list_(d, KEEP_PREFIX)
        assert [round(t, 6) for t, _ in periodic] == [0.005, 0.01, 0.02], periodic
        assert len(kept) == 1 and round(kept[0][0], 6) == 0.05, kept
        assert latest(d)[1] == periodic[-1][1]
        assert os.path.basename(periodic[1][1]) == "t_0.010000.npz"
        got, meta = load(periodic[1][1])
        assert np.array_equal(got, that)
        assert (meta["n"], meta["fac"]) == (7, 40.0)
        assert meta["keys"] == ["fac", "n", "t", "that"], meta["keys"]
        assert not any(f.endswith(".tmp.npz") for f in os.listdir(d)), \
            os.listdir(d)
        missing = os.path.join(d, "t_9.000000.npz")
        try:
            load(missing)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing file did not raise")
        bad = os.path.join(d, "bad.npz")
        np.savez_compressed(bad, x=1.0)
        try:
            load(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("file without 'that' did not raise")
    return True


if __name__ == "__main__":
    print("checkpoint I/O self-test: %s" % ("PASSED" if self_test() else "FAILED"))
