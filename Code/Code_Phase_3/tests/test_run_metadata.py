"""
The run-metadata JSON must always be writable.

Found in a parallel rehearsal: SnapshotAuditor.summary() returned a dict keyed
by (agent, model) tuples, so writing experiment_metadata.json raised
TypeError after every paid call had been made. Every real run exited
non-zero, and tools/merge_shards.py -- which requires that file as proof a
shard finished -- would have refused every shard. It stayed hidden because
it only fires once calls have actually been audited, which no offline test
did.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.snapshots import SnapshotAuditor  # noqa: E402
from src.store import write_json  # noqa: E402


def _populated():
    a = SnapshotAuditor()
    a.check("deepseek_primary", "deepseek", "deepseek-v4-flash", "focal", False)
    a.check("deepseek_primary", "deepseek", "deepseek-v4-flash", "focal", False)
    a.check("judge", "gemini", "gemini-2.5-flash", "judge", False)
    a.check("gpt4o_mini", "gpt-4o-mini", "gpt-4.1-mini", "focal", False)   # a mismatch
    return a


def test_summary_of_a_populated_auditor_is_json_serialisable():
    json.dumps(_populated().summary())


def test_summary_keeps_every_count_under_string_keys():
    s = _populated().summary()
    assert s["calls"] == 4
    assert s["mismatches"] == 1
    assert s["served_models"]["deepseek_primary"]["deepseek-v4-flash"] == 2
    assert s["served_models"]["judge"]["gemini-2.5-flash"] == 1
    assert all(isinstance(k, str) for k in s["served_models"])


def test_empty_auditor_summary_is_still_valid():
    assert SnapshotAuditor().summary() == {"calls": 0, "mismatches": 0, "served_models": {}}


def test_metadata_file_is_written_with_a_populated_auditor(tmp_path):
    """The exact write that crashed, end to end."""
    target = tmp_path / "experiment_metadata.json"
    write_json(target, {"snapshot_audit": _populated().summary(),
                        "shard": "deepseek_primary"})
    back = json.loads(target.read_text(encoding="utf-8"))
    assert back["snapshot_audit"]["calls"] == 4
