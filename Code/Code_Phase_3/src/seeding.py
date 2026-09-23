"""
seeding.py — process-stable seed derivation.

Phase-1/2 defect (next_plan.md §2.2): `trial_runner.py` derived per-trial seeds
with Python's builtin `hash(question_identifier)`. Since PEP 456, `hash()` on
str is salted per process (PYTHONHASHSEED), so peer ordering and persona-variant
choices were NOT reproducible across runs even though a master seed was fixed.

Everything here is a pure function of (master_seed, *string/int parts) via
zlib.crc32, which is stable across processes, platforms and Python versions.
"""

from __future__ import annotations

import zlib
from typing import Any, Iterable

# 2**31 - 1; keeps derived seeds inside the positive int32 range that numpy's
# legacy RandomState and random.Random both accept without surprises.
_MODULUS = 2_147_483_647


def stable_hash(*parts: Any) -> int:
    """
    Deterministic non-negative 32-bit hash of an arbitrary tuple of parts.

    Stable across processes and machines, unlike builtin hash() on str.
    """
    payload = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return zlib.crc32(payload) & 0xFFFFFFFF


def derive_seed(master_seed: int, *parts: Any) -> int:
    """
    Derive a child seed from the master seed and any number of identifying
    parts (question id, replicate index, condition, focal key, ...).

    Deterministic: the same arguments always give the same seed.
    """
    return (int(master_seed) + stable_hash(*parts)) % _MODULUS


def derive_rng(master_seed: int, *parts: Any):
    """Return a `random.Random` seeded by `derive_seed`."""
    import random

    return random.Random(derive_seed(master_seed, *parts))


def derive_numpy_rng(master_seed: int, *parts: Any):
    """Return a `numpy.random.Generator` seeded by `derive_seed`."""
    import numpy as np

    return np.random.default_rng(derive_seed(master_seed, *parts))


def unit_id(protocol: str, focal_key: str, condition: str, question_id: str,
            replicate: int, round_index: int = 1) -> str:
    """
    The canonical identifier for one measured unit.

    A *unit* is one (protocol, focal, condition, question, replicate, round)
    revision episode. The checkpoint file keys on this, so a crashed run
    resumes without repeating or duplicating any API call.
    """
    return f"{protocol}|{focal_key}|{condition}|{question_id}|r{replicate}|rd{round_index}"


def r0_unit_id(focal_key: str, question_id: str, replicate: int) -> str:
    """Canonical identifier for one cached Round-0 answer."""
    return f"R0|{focal_key}|{question_id}|r{replicate}"


def iter_stable_sorted(items: Iterable[Any]) -> list:
    """Sort any iterable of stringifiable items deterministically."""
    return sorted(items, key=lambda x: str(x))
