"""
snapshots.py — served-model assertion for focal calls.

Phase-2 defect (next_plan.md §2.6): the `gpt4o_mini` focal agent carried a
LinkAPI -> nano-gpt -> OpenRouter fallback chain. 75 of 15,000 focal responses
were served by `gpt-4.1-mini-2025-04-14` while being logged and reported as
GPT-4o-mini. The effect on reported numbers was <= 0.11 pp, but the paper
claimed a single checkpoint, so the claim was false as written.

Phase 3 policy:
  * every focal model declares `expected_served_prefix` in config/models.yaml;
  * every focal response is checked against it;
  * `strict=True` (the default for focal calls) ABORTS the run on mismatch;
  * every check is appended to results/outputs/snapshot_audit.parquet, so the
    appendix can state the served string distribution from the release itself.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("platos_ship3.snapshots")


class SnapshotMismatchError(RuntimeError):
    """Raised when a provider serves a model other than the pinned snapshot."""


@dataclass
class SnapshotAuditor:
    """Collects one row per model call; enforces the pin on focal calls."""

    rows: List[Dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def check(
        self,
        agent_key: str,
        expected_prefix: Optional[str],
        served_model: str,
        role: str,
        strict: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Record and (optionally) enforce one served-model observation.

        Returns True when the served string matches the pin. Raises
        SnapshotMismatchError when `strict` and it does not.
        """
        served = (served_model or "").strip()
        if not expected_prefix:
            ok = True
        else:
            ok = served.lower().startswith(expected_prefix.strip().lower())

        with self._lock:
            self.rows.append(
                {
                    "agent_key": agent_key,
                    "role": role,
                    "expected_served_prefix": expected_prefix or "",
                    "served_model": served,
                    "matched": bool(ok),
                    "strict": bool(strict),
                    **(context or {}),
                }
            )

        if not ok:
            message = (
                f"Snapshot mismatch for '{agent_key}' ({role}): expected a model "
                f"starting with '{expected_prefix}', provider served '{served}'. "
                f"Context: {context}"
            )
            if strict:
                logger.error(message)
                raise SnapshotMismatchError(message)
            logger.warning(message)
        return bool(ok)

    def to_frame(self):
        import pandas as pd

        with self._lock:
            return pd.DataFrame(self.rows)

    def save(self, path) -> None:
        """Write the audit to parquet (no-op when nothing was recorded)."""
        from pathlib import Path

        frame = self.to_frame()
        if frame.empty:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        logger.info(
            "Snapshot audit: %d calls, %d mismatches -> %s",
            len(frame),
            int((~frame["matched"]).sum()),
            path,
        )

    def summary(self) -> Dict[str, Any]:
        frame = self.to_frame()
        if frame.empty:
            return {"calls": 0, "mismatches": 0, "served_models": {}}
        return {
            "calls": int(len(frame)),
            "mismatches": int((~frame["matched"]).sum()),
            "served_models": (
                frame.groupby(["agent_key", "served_model"]).size().to_dict()
            ),
        }
