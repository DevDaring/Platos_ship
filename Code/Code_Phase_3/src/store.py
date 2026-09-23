"""
store.py — resume-capable incremental writes.

House rules (CLAUDE.md): never lose progress on a crash. Every runner appends
rows through an `IncrementalWriter`, which flushes a parquet shard every N rows
and records completed unit ids in a checkpoint file. Re-running any experiment
skips units already present, so a killed run resumes exactly where it stopped
without repeating a paid API call.

Shards are merged into the canonical parquet by `consolidate()`, which is
idempotent and de-duplicates on the unit id.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd

logger = logging.getLogger("platos_ship3.store")


class Checkpoint:
    """The set of unit ids already completed, persisted as a parquet column."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._ids: Set[str] = set()
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                frame = pd.read_parquet(self.path)
                self._ids = set(frame["unit_id"].astype(str))
                logger.info("Checkpoint: %d completed units in %s",
                            len(self._ids), self.path)
            except Exception as exc:
                logger.warning("Could not read checkpoint %s (%s); starting empty.",
                               self.path, exc)
                self._ids = set()

    def __contains__(self, unit_id: str) -> bool:
        with self._lock:
            return unit_id in self._ids

    def add(self, unit_id: str) -> None:
        with self._lock:
            self._ids.add(unit_id)

    def add_many(self, unit_ids: Iterable[str]) -> None:
        with self._lock:
            self._ids.update(unit_ids)

    def save(self) -> None:
        with self._lock:
            ids = sorted(self._ids)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"unit_id": ids}).to_parquet(self.path, index=False)
        logger.info("Checkpoint saved: %d units -> %s", len(ids), self.path)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._ids)


class IncrementalWriter:
    """
    Buffered appender that flushes parquet shards next to the target file.

    `target` is the canonical output (e.g. results/outputs/revision_log.parquet).
    Shards live in `<target_parent>/_shards/<target_stem>/part-XXXXX.parquet`.
    """

    def __init__(self, target: Path, flush_every: int = 200,
                 checkpoint: Optional[Checkpoint] = None):
        self.target = Path(target)
        self.flush_every = int(flush_every)
        self.checkpoint = checkpoint
        self.shard_dir = self.target.parent / "_shards" / self.target.stem
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: List[Dict[str, Any]] = []
        self._pending_ids: List[str] = []
        self._lock = threading.Lock()
        self._shard_index = self._next_shard_index()

    def _next_shard_index(self) -> int:
        existing = sorted(self.shard_dir.glob("part-*.parquet"))
        if not existing:
            return 0
        return max(int(p.stem.split("-")[-1]) for p in existing) + 1

    def append(self, row: Dict[str, Any], unit_id: Optional[str] = None) -> None:
        with self._lock:
            self._buffer.append(row)
            if unit_id:
                self._pending_ids.append(unit_id)
            should_flush = len(self._buffer) >= self.flush_every
        if should_flush:
            self.flush()

    def extend(self, rows: Iterable[Dict[str, Any]],
               unit_id: Optional[str] = None) -> None:
        rows = list(rows)
        if not rows:
            return
        with self._lock:
            self._buffer.extend(rows)
            if unit_id:
                self._pending_ids.append(unit_id)
            should_flush = len(self._buffer) >= self.flush_every
        if should_flush:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            buffer, ids = self._buffer, self._pending_ids
            self._buffer, self._pending_ids = [], []
            shard_index = self._shard_index
            self._shard_index += 1

        path = self.shard_dir / f"part-{shard_index:05d}.parquet"
        pd.DataFrame(buffer).to_parquet(path, index=False)
        if self.checkpoint is not None and ids:
            self.checkpoint.add_many(ids)
            self.checkpoint.save()
        logger.debug("Flushed %d rows -> %s", len(buffer), path)

    def consolidate(self, dedup_on: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Merge the existing target and every shard into one parquet.

        Idempotent: safe to call after each experiment. De-duplicates on
        `dedup_on` (keeping the last occurrence) when given.
        """
        self.flush()
        frames: List[pd.DataFrame] = []
        if self.target.exists():
            frames.append(pd.read_parquet(self.target))
        for shard in sorted(self.shard_dir.glob("part-*.parquet")):
            frames.append(pd.read_parquet(shard))

        if not frames:
            return pd.DataFrame()

        merged = pd.concat(frames, ignore_index=True)
        if dedup_on:
            present = [c for c in dedup_on if c in merged.columns]
            if present:
                merged = merged.drop_duplicates(subset=present, keep="last")
        self.target.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(self.target, index=False)

        for shard in self.shard_dir.glob("part-*.parquet"):
            shard.unlink()
        logger.info("Consolidated %d rows -> %s", len(merged), self.target)
        return merged


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON artefact, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def read_parquet_if_exists(path: Path) -> Optional[pd.DataFrame]:
    """Read a parquet file, or return None when it is absent."""
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_parquet(path)
