"""
concurrency.py — bounded, fail-fast parallelism inside one process.

One shard process runs one focal model. Its units (one (question,
replicate) cell of one condition) are independent: the Round-0 answer they
start from is already cached, their peers are drawn from a stream seeded on
the unit alone (src/peer_pools.peer_rng), and a multi-round unit runs its
rounds in order inside a single worker. So running several units at once
changes the wall-clock time and nothing else. What had to become safe for
that is shared state, and all of it is locked: the writers and checkpoints
(src/store.py), key rotation and usage counters (agent_wrappers), the
snapshot auditor, the judge's counters and the failure tally (call_guard).

Measured before choosing this (24 Sept 2026, 8 concurrent calls): nano-gpt
served Llama-3.1-8B at 16 s per call sequentially and 8 calls in 21 s; Qwen
about 5x; OpenRouter's Gemma-3-4B returned upstream 429s, so OpenRouter
models get fewer workers (tools/run_parallel.sh).

Fail fast: the first exception in any worker cancels every unit not yet
started and is re-raised, so a provenance violation (a wrong served model)
stops the shard exactly as it did when the loop was sequential.
"""

from __future__ import annotations

import logging
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from typing import Callable, Iterable, List, TypeVar

logger = logging.getLogger("platos_ship3.concurrency")

T = TypeVar("T")
R = TypeVar("R")


def run_units(items: Iterable[T], fn: Callable[[T], R], workers: int = 1,
              label: str = "units") -> List[R]:
    """Apply `fn` to every item on up to `workers` threads; fail fast."""
    items = list(items)
    if int(workers) <= 1 or len(items) <= 1:
        return [fn(item) for item in items]

    executor = ThreadPoolExecutor(max_workers=int(workers),
                                  thread_name_prefix=label[:20])
    futures = [executor.submit(fn, item) for item in items]
    try:
        done, _ = wait(futures, return_when=FIRST_EXCEPTION)
        for future in done:
            error = future.exception()
            if error is not None:
                raise error
        return [future.result() for future in futures]
    finally:
        # On an exception: drop every unit not yet started, let the ones in
        # flight finish their single call, then propagate.
        executor.shutdown(wait=True, cancel_futures=True)
