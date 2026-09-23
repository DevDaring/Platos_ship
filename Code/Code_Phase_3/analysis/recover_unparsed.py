"""
Recover unparsed Round-1 answers with the judge cascade.

Runs OFFLINE against a trials parquet that carries the raw generations. It
needs no GPU: the generations are already on disk and the judges are API
models. It will refuse to run on a parquet written before the probe was
fixed to persist `round1_text`, because there is then nothing to re-read.

  python3 analysis/recover_unparsed.py --trials <path> [--calibrate 150]

FOUR GUARDRAILS, BECAUSE A JUDGE CAN MANUFACTURE AN EFFECT.

1. CONDITION-BLIND. The judge is shown the question, the options and the
   response text. It is never shown the condition, the peer messages, the
   target or the round. A judge that knew which arm a trial came from could
   resolve ambiguous text in the direction of the hypothesis, and the
   recovered subset would then encode the hypothesis rather than test it.
   `_assert_blind` re-checks this on the built prompt for every call, so the
   property cannot be lost by a later edit to the prompt builder.

2. EXTRACTION, NEVER SOLVING. The judge reports the answer the response
   COMMITTED TO, and returns UNPARSEABLE when the response never commits.
   The correct answer is withheld for the same reason. A judge that solved
   the problem would replace a missing observation with its own competence.

3. ABSTENTION IS A RESULT. Generations are capped at 600 tokens and some
   run out mid-derivation, so for those trials no stated answer exists.
   Those stay unrecovered and keep flowing into the ambiguous set of
   `nonparse_bounds`, rather than being filled in.

4. CALIBRATED ON ROWS THE REGEX ALREADY READ. `--calibrate` re-judges a
   random sample of rows the regex parsed successfully and reports the
   agreement rate. A recovered answer is only worth as much as that number,
   so the run reports it rather than assuming it.

Writes are incremental: every batch is flushed, and a re-run skips work
already recorded, so an interrupted run resumes without re-paying for it.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("recover_unparsed")

TEXT_COLUMN = "round1_text"
ANSWER_COLUMN = "round1_answer"
JUDGED_COLUMN = "round1_answer_judged"
METHOD_COLUMN = "round1_extraction_method"
ABSTAIN = "UNPARSEABLE"

# Anything that would tell the judge which arm the trial came from.
_LEAKY = ("condition", "wrong_peer", "WR", "peer said", "the other agent",
          "fixed_target", "correct_answer", "round 0", "round 1")


def _assert_blind(prompt: str, row: Dict[str, Any]) -> None:
    """
    Fail loudly rather than silently contaminate the extraction.

    Checked per call, not once at start-up: the prompt builder lives in the
    judge wrapper and could acquire a new field at any time.
    """
    lowered = prompt.lower()
    for marker in _LEAKY:
        if marker.lower() in lowered:
            raise AssertionError(
                f"judge prompt leaks '{marker}'; the extraction would no "
                "longer be blind to the experimental arm"
            )
    for field in ("condition", "fixed_target", "correct_answer"):
        value = str(row.get(field) or "").strip()
        if value and len(value) > 2 and value.lower() in lowered:
            raise AssertionError(f"judge prompt leaks the value of {field}")


def _needs_recovery(frame: pd.DataFrame) -> pd.Series:
    """Blank regex answer, but text present for a judge to read."""
    blank = frame[ANSWER_COLUMN].astype(str).str.strip().eq("")
    has_text = frame[TEXT_COLUMN].astype(str).str.strip().ne("")
    return blank & has_text


def _options_for(row: Dict[str, Any]) -> str:
    options = row.get("answer_options")
    if options is None or (isinstance(options, float) and pd.isna(options)):
        return "a number"
    if isinstance(options, (list, tuple)):
        return ", ".join(str(o) for o in options)
    return str(options)


def recover(
    trials: pd.DataFrame,
    cascade: Any,
    questions: Optional[pd.DataFrame] = None,
    checkpoint: Optional[Path] = None,
    flush_every: int = 25,
) -> pd.DataFrame:
    """Judge every unparsed row. Returns the frame with two columns added."""
    frame = trials.copy()
    if TEXT_COLUMN not in frame.columns:
        raise ValueError(
            f"'{TEXT_COLUMN}' is absent. This parquet was written by the "
            "probe before it persisted raw generations, so the text the "
            "judge would read no longer exists. Re-run the probe with the "
            "current run_big_probe.py, or use analysis/nonparse_bounds.py, "
            "which bounds the estimand without needing the text."
        )

    if JUDGED_COLUMN not in frame.columns:
        frame[JUDGED_COLUMN] = ""
        frame[METHOD_COLUMN] = ""

    question_text: Dict[str, Dict[str, Any]] = {}
    if questions is not None:
        question_text = {
            str(r["question_identifier"]): r
            for r in questions.to_dict("records")
        }

    targets = frame.index[
        _needs_recovery(frame)
        & frame[METHOD_COLUMN].astype(str).str.strip().eq("")
    ].tolist()
    # Judged in random order so that any drift in a provider's behaviour
    # over the run cannot line up with condition, which is written in
    # blocks.
    random.Random(20260502).shuffle(targets)
    logger.info("rows needing recovery: %d", len(targets))

    done = 0
    for index in targets:
        row = frame.loc[index].to_dict()
        meta = question_text.get(str(row.get("question_identifier")), {})
        prompt = cascade._build_user_prompt(
            str(meta.get("question_text", "")),
            _options_for({**row, **meta}),
            str(row[TEXT_COLUMN]),
        )
        _assert_blind(prompt, row)
        try:
            answer, method = cascade.extract_answer(
                str(meta.get("question_text", "")),
                _options_for({**row, **meta}),
                str(row[TEXT_COLUMN]),
            )
        except Exception as exc:                      # keep the run alive
            logger.warning("judge failed on %s: %s", index, exc)
            answer, method = ABSTAIN, "judge_error"
        frame.at[index, JUDGED_COLUMN] = "" if answer == ABSTAIN else answer
        frame.at[index, METHOD_COLUMN] = method
        done += 1
        if checkpoint is not None and done % flush_every == 0:
            frame.to_parquet(checkpoint, index=False)
            logger.info("checkpoint at %d/%d", done, len(targets))
    if checkpoint is not None:
        frame.to_parquet(checkpoint, index=False)
    return frame


def calibrate(
    trials: pd.DataFrame,
    cascade: Any,
    questions: Optional[pd.DataFrame] = None,
    sample_size: int = 150,
    seed: int = 20260502,
) -> Dict[str, Any]:
    """
    Agreement between judge and regex on rows the REGEX ALREADY READ.

    This is the number that licenses the recovered answers. Reported, never
    assumed: a judge that disagrees with the regex on rows where the regex
    was confident is not a judge whose verdicts should enter the analysis.
    """
    parsed = trials[
        trials[ANSWER_COLUMN].astype(str).str.strip().ne("")
        & trials[TEXT_COLUMN].astype(str).str.strip().ne("")
    ]
    if parsed.empty:
        return {"n": 0, "agreement": float("nan")}
    sample = parsed.sample(n=min(sample_size, len(parsed)), random_state=seed)
    lookup = ({str(r["question_identifier"]): r
               for r in questions.to_dict("records")} if questions is not None
              else {})

    agree = 0
    disagreements: List[Dict[str, str]] = []
    for row in sample.to_dict("records"):
        meta = lookup.get(str(row.get("question_identifier")), {})
        try:
            answer, _ = cascade.extract_answer(
                str(meta.get("question_text", "")),
                _options_for({**row, **meta}),
                str(row[TEXT_COLUMN]),
            )
        except Exception:
            continue
        if str(answer).strip().upper() == str(row[ANSWER_COLUMN]).strip().upper():
            agree += 1
        elif len(disagreements) < 20:
            disagreements.append({"regex": str(row[ANSWER_COLUMN]),
                                  "judge": str(answer)})
    return {
        "n": int(len(sample)),
        "n_agree": agree,
        "agreement": agree / len(sample),
        "example_disagreements": disagreements,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", required=True, type=Path)
    parser.add_argument("--questions", type=Path,
                        default=_ROOT / "results/processed/question_pool.parquet")
    parser.add_argument("--models-config", type=Path,
                        default=_ROOT / "config/models.yaml")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--calibrate", type=int, default=150,
                        help="rows the regex already read, re-judged to "
                             "measure agreement; 0 disables")
    args = parser.parse_args()

    import yaml
    from src.agent_wrappers.judge_agent import JudgeCascade

    trials = pd.read_parquet(args.trials)
    if TEXT_COLUMN not in trials.columns:
        logger.error(
            "'%s' absent -- this parquet predates the fix that persists raw "
            "generations. Nothing to judge. Use analysis/nonparse_bounds.py.",
            TEXT_COLUMN)
        return 2

    questions = (pd.read_parquet(args.questions)
                 if args.questions.exists() else None)
    cascade = JudgeCascade(yaml.safe_load(args.models_config.read_text()))
    out = args.out or args.trials.with_name(
        args.trials.stem + "_judged.parquet")

    report: Dict[str, Any] = {}
    if args.calibrate:
        report["calibration"] = calibrate(
            trials, cascade, questions, sample_size=args.calibrate)
        logger.info("judge/regex agreement: %.4f on n=%d",
                    report["calibration"]["agreement"],
                    report["calibration"]["n"])

    recovered = recover(trials, cascade, questions, checkpoint=out)
    still_blank = int(
        recovered[JUDGED_COLUMN].astype(str).str.strip().eq("").sum()
        - (~_needs_recovery(trials)).sum())
    report["n_recovered"] = int(
        recovered[JUDGED_COLUMN].astype(str).str.strip().ne("").sum())
    report["n_abstained"] = max(still_blank, 0)
    report["judge_usage"] = cascade.usage_stats
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
