#!/usr/bin/env python3
"""
run_status.py — lightweight progress snapshot for the 15-minute auto-push.

Reads the live Phase-2 checkpoint (data/outputs/completed_trials.parquet) and
writes a small, human-readable status file plus a machine-readable JSON. These
are cheap to commit every 15 minutes; the 40 MB trial_log.parquet is not.

    python3 scripts/run_status.py [--phase2-root <dir>] [--out <dir>]
"""

import argparse
import datetime
import json
import subprocess
from pathlib import Path

import pandas as pd

# Target trial counts per (experiment, focal, condition) so progress is a
# percentage rather than a bare count. Populated from config/experiment.yaml.
DEFAULT_PHASE2 = Path(__file__).resolve().parent.parent / "Code_Phase_2" / "CPU_Only"


def _tail(path: Path, n: int = 12) -> list:
    if not path.exists():
        return []
    try:
        return path.read_text(errors="replace").splitlines()[-n:]
    except OSError:
        return []


def _gpu_probe_rows(phase2_root: Path) -> int:
    p = phase2_root.parent / "GPU_Only" / "data" / "outputs" / "logprob_probe_trials.parquet"
    if not p.exists():
        return 0
    try:
        return len(pd.read_parquet(p))
    except Exception:
        return 0


def build_status(phase2_root: Path) -> dict:
    ckpt = phase2_root / "data" / "outputs" / "completed_trials.parquet"
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    status = {
        "generated_utc": now,
        "checkpoint_exists": ckpt.exists(),
        "total_completed_trials": 0,
        "by_focal_and_condition": [],
        "gpu_probe_trials": _gpu_probe_rows(phase2_root),
        "recent_log_lines": _tail(phase2_root / "logs" / "phase2_run.log"),
        "recent_api_failures": _tail(phase2_root / "logs" / "api_failures.log", 6),
    }

    if ckpt.exists():
        df = pd.read_parquet(ckpt)
        status["total_completed_trials"] = int(len(df))
        grouped = (
            df.groupby(["focal_smart_agent_name", "condition_identifier"])
            .size()
            .reset_index(name="trials")
            .sort_values(["focal_smart_agent_name", "condition_identifier"])
        )
        status["by_focal_and_condition"] = grouped.to_dict(orient="records")

    return status


def render_markdown(status: dict) -> str:
    lines = [
        "# Run status",
        "",
        f"_Auto-generated {status['generated_utc']} by `scripts/run_status.py`; "
        "refreshed by the 15-minute auto-push._",
        "",
        f"**Total completed trials (Phase 2 checkpoint): {status['total_completed_trials']:,}**",
        "",
    ]
    if status["gpu_probe_trials"]:
        lines += [f"**GPU mechanistic-probe trials: {status['gpu_probe_trials']:,}**", ""]

    if status["by_focal_and_condition"]:
        lines += ["| Focal model | Condition | Trials |", "|---|---|---:|"]
        for r in status["by_focal_and_condition"]:
            lines.append(
                f"| {r['focal_smart_agent_name']} | {r['condition_identifier']} | {r['trials']:,} |"
            )
        lines.append("")

    if status["recent_log_lines"]:
        lines += ["## Last log lines", "", "```"] + status["recent_log_lines"] + ["```", ""]
    if status["recent_api_failures"]:
        lines += ["## Recent API failures", "", "```"] + status["recent_api_failures"] + ["```", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2-root", type=Path, default=DEFAULT_PHASE2)
    ap.add_argument("--out", type=Path, default=None,
                    help="Directory for RUN_STATUS.md / run_status.json "
                         "(default: <phase2-root>/../results/progress)")
    args = ap.parse_args()

    out = args.out or (args.phase2_root.parent / "results" / "progress")
    out.mkdir(parents=True, exist_ok=True)

    status = build_status(args.phase2_root)
    (out / "run_status.json").write_text(json.dumps(status, indent=2))
    (out / "RUN_STATUS.md").write_text(render_markdown(status))
    print(f"wrote {out}/RUN_STATUS.md  ({status['total_completed_trials']:,} trials)")


if __name__ == "__main__":
    main()
