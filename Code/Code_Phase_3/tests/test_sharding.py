"""
Tests for safe parallel execution (src/sharding.py, tools/merge_shards.py).

The claim under test: running one process per model produces exactly the
work a single sequential process would, with no file written by two
processes, and a merge that refuses anything it cannot prove equivalent.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import sharding  # noqa: E402
from src.agents import resolve_focal_selector  # noqa: E402

PATHS = yaml.safe_load((_ROOT / "config" / "paths.yaml").read_text(encoding="utf-8"))
EXPERIMENT = yaml.safe_load((_ROOT / "config" / "experiment.yaml").read_text(encoding="utf-8"))
MODELS = yaml.safe_load((_ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
FOCAL = MODELS["focal_agents"]


# -- classification and isolation ---------------------------------------
def test_every_paths_key_is_classified():
    """An unclassified output would be written by every process at once."""
    assert sharding.unclassified_keys(PATHS) == []


def test_every_run_output_is_redirected_and_no_input_moves():
    for shard in ("deepseek_primary", "sweep_gemma_3_27b"):
        sharded = sharding.shard_paths(PATHS, shard)
        sharding.assert_isolated(PATHS, sharded, shard)
        for key in sharding.RUN_OUTPUT_KEYS & set(PATHS):
            assert f"/shards/{shard}" in str(sharded[key]).replace("\\", "/"), key
        for key in sharding.SHARED_INPUT_KEYS & set(PATHS):
            assert sharded[key] == PATHS[key], key


def test_two_shards_share_no_output_path():
    a = sharding.shard_paths(PATHS, "model_a")
    b = sharding.shard_paths(PATHS, "model_b")
    outs_a = {str(a[k]) for k in sharding.RUN_OUTPUT_KEYS & set(PATHS)}
    outs_b = {str(b[k]) for k in sharding.RUN_OUTPUT_KEYS & set(PATHS)}
    assert outs_a.isdisjoint(outs_b)


def test_unclassified_key_refuses_shard_mode():
    with pytest.raises(sharding.ShardError, match="not classified"):
        sharding.shard_paths({**PATHS, "brand_new_output_file": "./x.parquet"}, "s")


@pytest.mark.parametrize("bad", ["", "../escape", "a/b", ".hidden", "a b"])
def test_unsafe_shard_names_are_refused(bad):
    with pytest.raises(sharding.ShardError):
        sharding.validate_shard_name(bad)


# -- the partition is exact against the real config ---------------------
def _pairs(focal_specs):
    out = set()
    for name, spec in EXPERIMENT["experiments"].items():
        if not spec.get("enabled") or spec.get("offline"):
            continue
        if name.startswith("X8"):
            continue
        for key in resolve_focal_selector(spec.get("focal"), focal_specs):
            out.add((name, key))
    return out


def test_one_process_per_model_reproduces_the_sequential_plan_exactly():
    """
    The core safety claim. Every (experiment, model) pair the sequential run
    would execute is executed by exactly one shard, and no shard executes a
    pair the sequential run would not.
    """
    sequential = _pairs(FOCAL)
    seen = {}
    for model in FOCAL:
        share = _pairs(sharding.restrict_focal_specs(FOCAL, [model]))
        assert all(k == model for _, k in share), (model, share)
        for pair in share:
            assert pair not in seen, f"{pair} ran in {seen.get(pair)} and {model}"
            seen[pair] = model
    assert set(seen) == sequential


def test_tier_selector_gives_a_model_only_its_own_experiments():
    only_4b = _pairs(sharding.restrict_focal_specs(FOCAL, ["sweep_gemma_3_4b_focal"]))
    assert {n for n, _ in only_4b} == {"X1_common_matrix"}
    deep = _pairs(sharding.restrict_focal_specs(FOCAL, ["deepseek_primary"]))
    assert {"X2_message_decomposition", "X3_scaling"} <= {n for n, _ in deep}


def test_unknown_focal_model_is_refused():
    with pytest.raises(sharding.ShardError, match="unknown focal"):
        sharding.restrict_focal_specs(FOCAL, ["not_a_model"])


# -- the lock -----------------------------------------------------------
def test_second_process_on_one_shard_is_refused(tmp_path):
    first = sharding.ShardLock(tmp_path / "s")
    first.acquire()
    try:
        with pytest.raises(sharding.ShardError, match="locked"):
            sharding.ShardLock(tmp_path / "s").acquire()
    finally:
        first.release()


def test_lock_is_reusable_after_release(tmp_path):
    with sharding.ShardLock(tmp_path / "s"):
        pass
    with sharding.ShardLock(tmp_path / "s"):
        pass


def test_stale_lock_from_a_dead_process_is_cleared(tmp_path):
    """A crashed run must not wedge its shard for ever."""
    root = tmp_path / "s"
    root.mkdir()
    import socket
    (root / ".shard.lock").write_text(
        json.dumps({"pid": 999_999_9, "host": socket.gethostname()}), encoding="utf-8")
    with sharding.ShardLock(root):
        pass


# -- input hashing -------------------------------------------------------
def test_input_hash_detects_a_content_change(tmp_path):
    f = tmp_path / "q.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(f)
    paths = {"question_pool_file": str(f)}
    before = sharding.hash_shared_inputs(paths, tmp_path)
    pd.DataFrame({"a": [1, 3]}).to_parquet(f)
    after = sharding.hash_shared_inputs(paths, tmp_path)
    assert before["question_pool_file"] != after["question_pool_file"]


# -- the merge -----------------------------------------------------------
from tools import merge_shards  # noqa: E402


def _make_shard(parent: Path, name: str, focal: str, units, hashes=None,
                lock=False, leftover=False, dry=False):
    s = parent / name
    s.mkdir(parents=True)
    pd.DataFrame({"unit_id": units, "focal_key": focal}).to_parquet(
        s / "revision_log.parquet")
    pd.DataFrame({"unit_id": units}).to_parquet(s / "completed_units.parquet")
    (s / "experiment_metadata.json").write_text(json.dumps({
        "focal_filter": [focal], "dry_run": dry,
        "shared_input_sha256": hashes or {"question_pool_file": "abc"}}))
    if lock:
        (s / ".shard.lock").write_text("{}")
    if leftover:
        (s / "_shards" / "revision_log").mkdir(parents=True)
        pd.DataFrame({"unit_id": ["x"]}).to_parquet(
            s / "_shards" / "revision_log" / "part-00000.parquet")
    return s


def test_disjoint_shards_verify_and_merge(tmp_path):
    a = _make_shard(tmp_path, "a", "m1", ["u1", "u2"])
    b = _make_shard(tmp_path, "b", "m2", ["u3"])
    info = merge_shards.verify([a, b])
    assert set(info["models"]) == {"m1", "m2"}
    merged = merge_shards.merge_output([a, b], "revision_log.parquet", ["unit_id"])
    assert sorted(merged.unit_id) == ["u1", "u2", "u3"]


def test_a_unit_in_two_shards_is_refused(tmp_path):
    a = _make_shard(tmp_path, "a", "m1", ["u1", "u2"])
    b = _make_shard(tmp_path, "b", "m2", ["u2"])
    with pytest.raises(merge_shards.MergeRefused, match="more than one shard"):
        merge_shards.merge_output([a, b], "revision_log.parquet", ["unit_id"])


def test_resume_duplicates_inside_one_shard_are_collapsed(tmp_path):
    a = _make_shard(tmp_path, "a", "m1", ["u1", "u1", "u2"])
    merged = merge_shards.merge_output([a], "revision_log.parquet", ["unit_id"])
    assert sorted(merged.unit_id) == ["u1", "u2"]


def test_different_inputs_are_refused(tmp_path):
    a = _make_shard(tmp_path, "a", "m1", ["u1"], {"question_pool_file": "abc"})
    b = _make_shard(tmp_path, "b", "m2", ["u2"], {"question_pool_file": "XYZ"})
    with pytest.raises(merge_shards.MergeRefused, match="differs"):
        merge_shards.verify([a, b])


def test_a_model_in_two_shards_is_refused(tmp_path):
    a = _make_shard(tmp_path, "a", "m1", ["u1"])
    b = _make_shard(tmp_path, "b", "m1", ["u2"])
    with pytest.raises(merge_shards.MergeRefused, match="ran in both"):
        merge_shards.verify([a, b])


def test_a_running_shard_is_refused(tmp_path):
    a = _make_shard(tmp_path, "a", "m1", ["u1"], lock=True)
    with pytest.raises(merge_shards.MergeRefused, match="still locked"):
        merge_shards.verify([a])


def test_a_crashed_shard_with_unmerged_parts_is_refused(tmp_path):
    a = _make_shard(tmp_path, "a", "m1", ["u1"], leftover=True)
    with pytest.raises(merge_shards.MergeRefused, match="unmerged part-files"):
        merge_shards.verify([a])


def test_mixing_dry_and_real_runs_is_refused(tmp_path):
    a = _make_shard(tmp_path, "a", "m1", ["u1"], dry=True)
    b = _make_shard(tmp_path, "b", "m2", ["u2"], dry=False)
    with pytest.raises(merge_shards.MergeRefused, match="dry-run"):
        merge_shards.verify([a, b])


def test_a_shard_that_never_finished_is_refused(tmp_path):
    a = _make_shard(tmp_path, "a", "m1", ["u1"])
    (a / "experiment_metadata.json").unlink()
    with pytest.raises(merge_shards.MergeRefused, match="never"):
        merge_shards.verify([a])
