"""
call_guard.py — never record a failed API call as data.

A call that fails after its retries returns `raw_text_output=""` with
`error_status="failure"`, and names the model it was pinned to as the served
model, so the provenance check passes. Before this module, every writer saved
that row and checkpointed its unit:

  * a failed Round-0 put an empty "your previous response" into EVERY
    condition for that (model, question, replicate);
  * a failed revision was graded as a wrong answer, and because conditions
    run one after another, an outage landed inside a single condition and
    appeared there as harmful flips;
  * a failed weak-model call left an empty honest-bank message that still
    counted toward H's two peers.

Because the unit was checkpointed, a resumed run never retried it.

The rule: a FAILED call is not written and not checkpointed, so the next run
of the same command retries it. An empty response with a success status is
NOT a failure: it is the model's own output (the Mistral X8 probe produced 18
such generations), and retrying it would quietly select against it.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict

logger = logging.getLogger("platos_ship3.call_guard")

_lock = threading.Lock()
_failures: Dict[str, int] = {}


def is_failed_call(response: Any) -> bool:
    return str(getattr(response, "error_status", "") or "") == "failure"


def record_failure(stage: str, unit: str, response: Any) -> None:
    with _lock:
        _failures[stage] = _failures.get(stage, 0) + 1
    logger.warning("%s: call FAILED for %s after %s retries; not recorded, "
                   "will be retried on the next run", stage, unit,
                   getattr(response, "retry_attempts_used", "?"))


def failure_counts() -> Dict[str, int]:
    with _lock:
        return dict(_failures)


def reset_failures() -> None:
    with _lock:
        _failures.clear()


def drop_failed_calls(frame, label: str):
    """
    Remove rows whose call failed at the provider. With the writers guarded
    there should be none; this is the second line of defence for a log
    written before the guard, and it says so loudly when it fires.
    """
    if frame is None or frame.empty or "error_status" not in frame.columns:
        return frame
    failed = frame["error_status"].astype(str).eq("failure")
    if failed.any():
        logger.warning("%s: %d rows from FAILED calls excluded from analysis.",
                       label, int(failed.sum()))
    return frame[~failed]
