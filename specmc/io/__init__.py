"""`specmc.io`: run-tree input/output.

    specmc.io.checkpoint    atomic checkpoint write / list / read

The frozen reference file layout is unchanged: a checkpoint is still
``<SPECMC_OUT>/<tag>_ckpt/{t,keep}_<t:%.6f>.npz``.  Only the code that writes and
reads it has one home instead of one per driver.
"""
from __future__ import annotations

from . import checkpoint

__all__ = ["checkpoint"]
