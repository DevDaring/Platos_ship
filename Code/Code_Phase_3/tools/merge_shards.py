"""
merge_shards.py — combine per-model shards into the canonical outputs.

  python3 tools/merge_shards.py            # verify, then merge
  python3 tools/merge_shards.py --check    # verify only, write nothing

Run once, after every shard process has exited, and before --analyse.

It REFUSES to write anything unless every check below passes. A refusal is
the correct outcome whenever the parallel run cannot be shown to equal the
run a single sequential process would have produced.

  1. No shard is still running.       A live lock means a writer is active.
  2. No shard left unmerged parts.    A crash leaves part-files behind; the
                                      shard must be resumed, not half-merged.
  3. Every shard read identical       Each shard records the SHA-256 of every
     inputs.                          shared input. One mismatch means the
                                      shards did not run the same experiment.
  4. No model ran in two shards.      One model per process is what keeps its
                                      cached Round-0 answer shared by all of
                                      its conditions.
  5. No unit appears in two shards.   Checked on each output's own key,
                                      imported from the module that writes it.
  6. Canonical outputs are not        Existing results are never overwritten
     overwritten without --force.     silently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.r0_cache import R0_DEDUP_KEYS  # noqa: E402
from src.revision_runner import REVISION_DEDUP_KEYS  # noqa: E402
from src.sharding import SHARD_PARENT  # noqa: E402
from src.verifier import VERIFIER_DEDUP_KEYS  # noqa: E402

PEER_KEYS = ["unit_id", "peer_display_name", "shown_to_focal"]

# paths.yaml key -> the key that identifies one unit in that output.
# None: a log with no unique key, concatenated with the shard attached.
MERGE_KEYS: Dict[str, Optional[List[str]]] = {
    "r0_cache_file": R0_DEDUP_KEYS,
    "revision_log_file": REVISION_DEDUP_KEYS,
    "peer_message_log_file": PEER_KEYS,
    "completed_units_checkpoint_file": ["unit_id"],
    "safeguard_file": VERIFIER_DEDUP_KEYS,
    "snapshot_audit_file": None,
}


class MergeRefused(RuntimeError):
    pass


def _resolve(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (ROOT / raw).resolve()


def discover(shard_parent: Path) -> List[Path]:
    if not shard_parent.exists():
        return []
    return sorted(p for p in shard_parent.iterdir()
                  if p.is_dir() and not p.name.startswith("."))


def load_metadata(shard: Path) -> Dict[str, Any]:
    meta = shard / "experiment_metadata.json"
    if not meta.exists():
        raise MergeRefused(
            f"shard {shard.name} has no experiment_metadata.json: it never "
            "finished. Resume it before merging.")
    return json.loads(meta.read_text(encoding="utf-8"))


def verify(shards: List[Path], accept_changed: Optional[List[str]] = None) -> Dict[str, Any]:
    if not shards:
        raise MergeRefused(f"no shards found under {SHARD_PARENT}")
    problems: List[str] = []
    metas: Dict[str, Dict[str, Any]] = {}

    for shard in shards:
        if (shard / ".shard.lock").exists():
            problems.append(f"{shard.name}: still locked, a writer may be active")
        leftovers = sorted((shard / "_shards").rglob("part-*.parquet")) \
            if (shard / "_shards").exists() else []
        if leftovers:
            problems.append(f"{shard.name}: {len(leftovers)} unmerged part-files "
                            "(crashed mid-run; resume it)")
        try:
            metas[shard.name] = load_metadata(shard)
        except MergeRefused as exc:
            problems.append(str(exc))

    # 3. identical inputs
    hashes = {name: m.get("shared_input_sha256") or {} for name, m in metas.items()}
    if hashes:
        reference_name = sorted(hashes)[0]
        reference = hashes[reference_name]
        for name, h in hashes.items():
            for key in sorted(set(reference) | set(h)):
                # An input named with --accept-changed-input may differ, and
                # the difference is written to the manifest. Used once: the
                # hedged pool was cleaned after the first merge, and the one
                # shard re-run later (Llama-3.1-70B) does not read it.
                if key in (accept_changed or []):
                    continue
                if reference.get(key) != h.get(key):
                    problems.append(
                        f"input {key} differs: {reference_name}={str(reference.get(key))[:12]} "
                        f"vs {name}={str(h.get(key))[:12]}")

    # 4. one model per shard, none repeated
    owner: Dict[str, str] = {}
    for name, m in metas.items():
        for focal in m.get("focal_filter") or []:
            if focal in owner:
                problems.append(f"model {focal} ran in both {owner[focal]} and {name}")
            owner[focal] = name
    # A shard whose last run left failed calls unrecorded is incomplete; its
    # launcher re-runs it until this is empty (src/call_guard.py).
    for name, meta in metas.items():
        failed = meta.get("failed_calls_not_recorded") or {}
        if failed:
            problems.append(f"{name}: incomplete, failed calls not yet retried {failed}")
    limits = {name: m.get("max_questions") for name, m in metas.items()}
    if len(set(limits.values())) > 1:
        problems.append(f"shards ran on different question limits: {limits}")
    dry = {name: bool(m.get("dry_run")) for name, m in metas.items()}
    if len(set(dry.values())) > 1:
        problems.append(f"shards mix dry-run and real runs: {dry}")

    if problems:
        raise MergeRefused("merge refused:\n  - " + "\n  - ".join(problems))
    return {"metadata": metas, "models": owner, "dry_run": any(dry.values())}


def merge_output(shards: List[Path], filename: str,
                 key: Optional[List[str]]) -> Optional[pd.DataFrame]:
    frames = []
    for shard in shards:
        path = shard / filename
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if key:
            present = [k for k in key if k in frame.columns]
            if present:
                frame = frame.drop_duplicates(subset=present, keep="last")
        frame = frame.assign(_shard=shard.name)
        frames.append(frame)
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    if key:
        present = [k for k in key if k in merged.columns]
        if present:
            dup = merged[merged.duplicated(subset=present, keep=False)]
            if not dup.empty:
                where = dup.groupby(present, dropna=False)["_shard"].unique().head(5)
                raise MergeRefused(
                    f"{filename}: {len(dup)} rows whose unit appears in more than "
                    f"one shard, e.g. {where.to_dict()}")
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify only")
    parser.add_argument("--accept-changed-input", action="append", default=[],
                        help="a shared input key allowed to differ between shards "
                             "(recorded in the manifest)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing canonical outputs")
    args = parser.parse_args()

    paths = yaml.safe_load((ROOT / "config" / "paths.yaml").read_text(encoding="utf-8"))
    shards = discover(ROOT / SHARD_PARENT)
    try:
        info = verify(shards, args.accept_changed_input)
    except MergeRefused as exc:
        print(exc)
        return 1
    print(f"verified {len(shards)} shards: {sorted(info['models'])}")

    merged: Dict[str, pd.DataFrame] = {}
    try:
        for key, unit_key in MERGE_KEYS.items():
            frame = merge_output(shards, Path(paths[key]).name, unit_key)
            if frame is not None:
                merged[key] = frame
                print(f"  {Path(paths[key]).name:<32} {len(frame):>7} rows")
    except MergeRefused as exc:
        print(f"merge refused:\n  - {exc}")
        return 1

    if args.check:
        print("check passed; nothing written (--check)")
        return 0

    targets = {key: _resolve(paths[key]) for key in merged}
    clobber = [str(t) for t in targets.values() if t.exists()]
    if clobber and not args.force:
        print("refusing to overwrite existing canonical outputs "
              "(pass --force if that is intended):\n  " + "\n  ".join(clobber))
        return 1

    for key, frame in merged.items():
        targets[key].parent.mkdir(parents=True, exist_ok=True)
        frame.drop(columns=["_shard"]).to_parquet(targets[key], index=False)

    manifest = {
        "shards": {name: {"focal_filter": m.get("focal_filter"),
                          "experiments_run": m.get("experiments_run"),
                          "unavailable_focal_models": m.get("unavailable_focal_models")}
                   for name, m in info["metadata"].items()},
        "models": info["models"],
        "rows": {Path(paths[k]).name: int(len(f)) for k, f in merged.items()},
        "shared_input_sha256": next(iter(info["metadata"].values())).get("shared_input_sha256"),
        "dry_run": info["dry_run"],
        "accepted_changed_inputs": {
            key: {name: (m.get("shared_input_sha256") or {}).get(key)
                  for name, m in info["metadata"].items()}
            for key in args.accept_changed_input},
        "merged_at_utc": pd.Timestamp.now("UTC").isoformat(),
    }
    meta_target = _resolve(paths["experiment_metadata_file"])
    combined = {"merged_from_shards": manifest,
                "per_shard": info["metadata"]}
    meta_target.write_text(json.dumps(combined, indent=2, default=str), encoding="utf-8")
    (meta_target.parent / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"merged -> canonical outputs; manifest at {meta_target.parent / 'merge_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
