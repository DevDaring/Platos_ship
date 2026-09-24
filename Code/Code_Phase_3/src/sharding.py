"""
sharding.py — safe multi-process execution, one process per focal model.

WHY ONE PROCESS PER MODEL, AND NEVER FINER

The runner makes one API call at a time, so all eight focal models run
back to back: about 61 hours. The models bill to different providers and
share no rate limit, so running them side by side cuts that to the time of
the busiest single model, about 12 to 15 hours.

A model must never be split across processes. Each model's Round-0 answer
is cached once and reused by every one of its conditions, which is what
makes the conditions comparable. Two processes serving one model would each
generate their own Round-0 answers, and its experiments would no longer be
conditioned on the same first answer.

WHY EVERY WRITE IS PRIVATE

`store.py` guards its files with `threading.Lock`, which does nothing
between processes. Shared across processes, those files lose data three
ways: two writers pick the same shard number and one overwrites the other;
each process saves its whole checkpoint, so the last save erases the
others' progress; and consolidation deletes shards another process wrote
after it listed the folder. The only safe rule is that no file is ever
written by more than one process, so each shard gets a private output root.

WHY THE CLASSIFICATION IS EXHAUSTIVE

Every key in paths.yaml must be classified below. A key that is not refuses
shard mode outright. Listing only the outputs to redirect would silently
leave a newly added output shared, which is exactly the class of bug this
module exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Read by every shard, written by none. Built once, before the parallel phase.
SHARED_INPUT_KEYS = frozenset({
    "data_root", "processed_data_directory",
    "question_pool_file", "anchored_personas_file",
    "correct_anchored_personas_file", "confidence_personas_file",
    "inputs_manifest_file",
    "hedged_personas_file", "honest_bank_file",
    "gsm_symbolic_pool_file", "gsm_symbolic_personas_file",
    "legacy_phase1_trial_log", "legacy_phase2_trial_log",
    "legacy_gpu_probe_trials",
    "huggingface_cache_directory",
})

# Written during a run. Redirected into the shard's private root.
RUN_OUTPUT_KEYS = frozenset({
    "output_directory", "logs_directory",
    "r0_cache_file", "revision_log_file", "peer_message_log_file",
    "completed_units_checkpoint_file", "experiment_metadata_file",
    "snapshot_audit_file", "safeguard_file",
})

# Written only by analysis, which never runs inside a shard.
ANALYSIS_OUTPUT_KEYS = frozenset({
    "tables_directory", "figures_directory",
    "registry_file", "metrics_file", "contrasts_file", "voting_file",
    "gate_report_file", "paper_numbers_file",
})

# The shared inputs whose content every shard must agree on. Hashed at start.
HASHED_INPUT_KEYS = (
    "question_pool_file", "anchored_personas_file",
    "correct_anchored_personas_file", "confidence_personas_file",
    "hedged_personas_file", "honest_bank_file",
    "gsm_symbolic_pool_file", "gsm_symbolic_personas_file",
)

SHARD_PARENT = "results/outputs/shards"
LOG_PARENT = "logs/shards"


class ShardError(RuntimeError):
    """A shard refused to run because running it would not be safe."""


def unclassified_keys(paths: Dict[str, Any]) -> List[str]:
    known = SHARED_INPUT_KEYS | RUN_OUTPUT_KEYS | ANALYSIS_OUTPUT_KEYS
    return sorted(k for k in paths if k not in known)


def validate_shard_name(name: str) -> str:
    if not name or not all(c.isalnum() or c in "_-." for c in name) \
            or name.startswith("."):
        raise ShardError(f"shard name {name!r} must be alphanumeric, '_', '-' or '.'")
    return name


def shard_paths(paths: Dict[str, Any], shard: str) -> Dict[str, Any]:
    """
    A copy of `paths` with every run output moved under the shard's root.

    Shared inputs are left exactly as they are, so every shard reads the same
    files. Refuses if any key is unclassified.
    """
    validate_shard_name(shard)
    missing = unclassified_keys(paths)
    if missing:
        raise ShardError(
            "paths.yaml has keys not classified in src/sharding.py: "
            f"{missing}. Classify each as a shared input, a run output or an "
            "analysis output before running in shard mode; an unclassified "
            "output would be written by every process at once.")
    root = f"./{SHARD_PARENT}/{shard}"
    out = dict(paths)
    for key in RUN_OUTPUT_KEYS:
        if key not in paths:
            continue
        if key == "output_directory":
            out[key] = root
        elif key == "logs_directory":
            out[key] = f"./{LOG_PARENT}/{shard}"
        else:
            out[key] = f"{root}/{Path(str(paths[key])).name}"
    return out


def assert_isolated(original: Dict[str, Any], sharded: Dict[str, Any],
                    shard: str) -> None:
    """Belt and braces: prove the redirect before anything is written."""
    for key in RUN_OUTPUT_KEYS & set(original):
        value = str(sharded[key]).replace("\\", "/")
        if f"/shards/{shard}" not in value:
            raise ShardError(f"run output {key} not isolated: {value}")
    for key in SHARED_INPUT_KEYS & set(original):
        if sharded[key] != original[key]:
            raise ShardError(f"shared input {key} was altered in shard mode")


def missing_shared_inputs(paths: Dict[str, Any], project_root: Path,
                          required: Iterable[str]) -> List[str]:
    missing = []
    for key in required:
        raw = paths.get(key)
        if raw is None:
            missing.append(f"{key} (not in paths.yaml)")
            continue
        path = Path(raw)
        path = path if path.is_absolute() else (project_root / raw).resolve()
        if not path.exists():
            missing.append(f"{key} -> {path}")
    return missing


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_shared_inputs(paths: Dict[str, Any], project_root: Path) -> Dict[str, str]:
    """SHA-256 of each shared input that exists. Absent ones are recorded as such."""
    out: Dict[str, str] = {}
    for key in HASHED_INPUT_KEYS:
        raw = paths.get(key)
        if raw is None:
            continue
        path = Path(raw)
        path = path if path.is_absolute() else (project_root / raw).resolve()
        out[key] = sha256_file(path) if path.exists() else "ABSENT"
    return out


class ShardLock:
    """
    Exclusive lock on a shard's root, taken atomically with O_CREAT|O_EXCL.

    Two processes pointed at one shard would reintroduce every shared-file
    hazard this module removes. A lock left by a process that has died is
    detected by its PID and cleared, so a crash does not wedge the shard.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / ".shard.lock"
        self._held = False

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def acquire(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"pid": os.getpid(), "host": socket.gethostname()})
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    held = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    held = {}
                same_host = held.get("host") == socket.gethostname()
                if same_host and not self._pid_alive(int(held.get("pid", 0))):
                    self.path.unlink(missing_ok=True)   # stale: owner is dead
                    continue
                raise ShardError(
                    f"shard {self.root.name} is locked by pid {held.get('pid')} "
                    f"on {held.get('host')}; two processes must never share a "
                    "shard") from None
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            self._held = True
            return
        raise ShardError(f"could not acquire the lock for {self.root}")

    def release(self) -> None:
        if self._held:
            self.path.unlink(missing_ok=True)
            self._held = False

    def __enter__(self) -> "ShardLock":
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


def restrict_focal_specs(focal_specs: Dict[str, Any],
                         keep: Optional[Iterable[str]]) -> Dict[str, Any]:
    """
    Keep only the named focal models.

    Resolving an experiment's selector against the restricted specs gives
    exactly this shard's share of it: "ALL" becomes just these models, and a
    tier such as "TIER_X2" becomes these models only if they belong to it.
    """
    if keep is None:
        return dict(focal_specs)
    keep = list(keep)
    unknown = [k for k in keep if k not in focal_specs]
    if unknown:
        raise ShardError(f"unknown focal model(s): {unknown}; "
                         f"known: {sorted(focal_specs)}")
    return {k: v for k, v in focal_specs.items() if k in keep}
