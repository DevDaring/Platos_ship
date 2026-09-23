"""
Recover unparsed answers with the judge cascade, for BOTH rounds.

Runs OFFLINE against a trials parquet that carries the raw generations. It
needs no GPU: the generations are already on disk and the judges are API
models. It refuses to run on a parquet written before the probe was fixed
to persist the text, because there is then nothing to re-read.

  python3 analysis/recover_unparsed.py --trials <path> [--calibrate 150]

Both rounds are recovered, not just Round 1. The stated-answer-unchanged
subset requires an answer in BOTH rounds, so a Round-0 failure removes a
trial just as surely as a Round-1 failure does.

FIVE GUARDRAILS, BECAUSE A JUDGE CAN MANUFACTURE AN EFFECT.

1. CONDITION-BLIND. The judge sees the question, the options and the
   response text. It never sees the condition, the peer messages, the
   target or the round. A judge that knew which arm a trial came from could
   resolve ambiguous text in the direction of the hypothesis, and the
   recovered subset would then encode the hypothesis rather than test it.
   `_assert_blind` re-checks this on the built prompt for every call, so
   the property cannot be lost by a later edit to the prompt builder.

2. EXTRACTION, NEVER SOLVING. The judge reports the answer the response
   COMMITTED TO. The correct answer is withheld. This is enforced in
   `judge_agent.py` by a mechanical check that the returned answer occurs
   in the response text, because prompt wording alone did not stop a tier
   from computing 8 x 7 for a response that only said it needed to think
   about it.

3. ABSTENTION IS A RESULT, NOT A TIER FAILURE. Generations are capped at
   600 tokens and some run out mid-derivation, so for those trials no
   stated answer exists. They stay unrecovered and keep flowing into the
   ambiguous set of `nonparse_bounds`, rather than being filled in.

4. CALIBRATED ON ROWS THE REGEX ALREADY READ. `--calibrate` re-judges a
   random sample of rows the regex parsed and reports the agreement rate.
   A recovered answer is worth exactly what that number says it is.

5. ORDER IS RANDOMISED. Conditions are written to the parquet in blocks,
   so judging in file order would let any drift in a provider's behaviour
   line up with condition. The work queue is shuffled under a fixed seed.

Writes are incremental: every batch is flushed and a re-run skips work
already recorded, so an interrupted run resumes without re-paying for it.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("recover_unparsed")

ROUNDS: Tuple[str, ...] = ("round0", "round1")
ABSTAIN = "UNPARSEABLE"

# Anything that would tell the judge which arm the trial came from.
_LEAKY = ("condition", "wrong_peer", "peer said", "the other agent",
          "fixed_target", "correct_answer", "round 0", "round 1")


def text_column(round_name: str) -> str:
    return f"{round_name}_text"


def answer_column(round_name: str) -> str:
    return f"{round_name}_answer"


def judged_column(round_name: str) -> str:
    return f"{round_name}_answer_judged"


def method_column(round_name: str) -> str:
    return f"{round_name}_extraction_method"


def effective_column(round_name: str) -> str:
    return f"{round_name}_answer_effective"


_SENTINELS = ("\x00QUESTION\x00", "\x00OPTIONS\x00", "\x00RESPONSE\x00")


def _assert_blind(cascade: Any, prompt: str, row: Dict[str, Any],
                  question: str = "", options: str = "",
                  raw_text: str = "") -> None:
    """
    The prompt must be the builder's template filled with the three blind
    inputs, and nothing else.

    THREE EARLIER VERSIONS OF THIS CHECK WERE WRONG, each aborting a run on
    a false positive:

      1. It scanned the focal model's own response, so a response that
         answered correctly was flagged for containing correct_answer.
      2. It scanned for the word "condition", which is ordinary English
         and appears in real exam questions.
      3. It tried to recover the template by subtracting the three inputs
         from the prompt in sequence. When the response quoted the
         question, removing the question first altered the response, so the
         response no longer matched itself and was never removed. The
         model's words stayed in what the check believed was the template,
         and "the other agent" tripped it.

    Subtraction cannot work, because the inputs can contain one another.
    The template is therefore obtained from the BUILDER, by calling it with
    sentinels no natural text contains. That gives the template exactly,
    with no content in it at all, and proves the prompt is that template
    filled with these inputs and nothing more.
    """
    skeleton = cascade._build_user_prompt(*_SENTINELS)
    expected = skeleton
    for sentinel, value in zip(_SENTINELS, (question, options, raw_text)):
        expected = expected.replace(sentinel, str(value or ""))
    if prompt != expected:
        raise AssertionError(
            "judge prompt is not the question/options/response template; "
            "something else was interpolated into it")

    template = skeleton
    for sentinel in _SENTINELS:
        template = template.replace(sentinel, " ")
    lowered = template.lower()

    for marker in _LEAKY:
        if marker.lower() in lowered:
            raise AssertionError(
                f"judge prompt template leaks '{marker}'; the extraction "
                "would no longer be blind to the experimental arm"
            )
    for field in ("condition", "fixed_target", "correct_answer"):
        value = str(row.get(field) or "").strip()
        if value and len(value) > 2 and value.lower() in lowered:
            raise AssertionError(
                f"judge prompt template leaks the value of {field}")


def _needs_recovery(frame: pd.DataFrame, round_name: str) -> pd.Series:
    """Blank extracted answer, but text present for a judge to read."""
    blank = frame[answer_column(round_name)].astype(str).str.strip().eq("")
    has_text = frame[text_column(round_name)].astype(str).str.strip().ne("")
    return blank & has_text


def _options_for(row: Dict[str, Any]) -> str:
    """
    Multiple-choice options, LABELLED WITH THE LETTERS THE MODEL USED.

    Passing the bare option texts cost 33 points of judge/regex agreement.
    The focal model answers "Final answer: G", but the judge was shown only
    ["42", "0", "15", ...] with no letters, so "G" matched nothing it had
    been told was valid. It then either abstained on a perfectly clear
    answer or returned the option TEXT ("-42") where the regex returned the
    letter ("G") -- right about the content, unable to say it in the same
    alphabet.

    Option order is the letter order: index 0 is A, as the probe's own
    extractor assumes.
    """
    options = row.get("answer_options")
    if options is None or (isinstance(options, float) and pd.isna(options)):
        return "a number"
    listed: Optional[List[Any]] = None
    if isinstance(options, (list, tuple)):
        listed = list(options)
    elif hasattr(options, "tolist"):
        try:
            listed = list(options.tolist())
        except Exception:
            listed = None
    elif isinstance(options, str):
        # The pool stores the options as a JSON string, not a list. Missing
        # this is what made the labelling fix a no-op the first time: the
        # function fell through and handed the judge the raw JSON.
        text = options.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    listed = parsed
            except (ValueError, TypeError):
                listed = None
    if listed:
        letters = [f"{chr(ord('A') + i)}) {o}" for i, o in enumerate(listed)]
        return ("; ".join(str(x) for x in letters)
                + "  -- answer with the single capital letter")
    return str(options)


def _require_text(frame: pd.DataFrame, rounds: Sequence[str]) -> None:
    missing = [text_column(r) for r in rounds if text_column(r) not in frame.columns]
    if missing:
        raise ValueError(
            f"{', '.join(missing)} absent. This parquet was written by the "
            "probe before it persisted raw generations, so the text the "
            "judge would read no longer exists. Re-run the probe with the "
            "current run_big_probe.py, or use analysis/nonparse_bounds.py, "
            "which bounds the estimand without needing the text."
        )


def recover(
    trials: pd.DataFrame,
    cascade: Any,
    questions: Optional[pd.DataFrame] = None,
    checkpoint: Optional[Path] = None,
    flush_every: int = 25,
    rounds: Sequence[str] = ROUNDS,
) -> pd.DataFrame:
    """Judge every unparsed row in every requested round."""
    frame = trials.copy()
    _require_text(frame, rounds)

    for round_name in rounds:
        if judged_column(round_name) not in frame.columns:
            frame[judged_column(round_name)] = ""
            frame[method_column(round_name)] = ""

    lookup: Dict[str, Dict[str, Any]] = {}
    if questions is not None:
        lookup = {str(r["question_identifier"]): r
                  for r in questions.to_dict("records")}

    queue: List[Tuple[Any, str]] = []
    for round_name in rounds:
        pending = frame.index[
            _needs_recovery(frame, round_name)
            & frame[method_column(round_name)].astype(str).str.strip().eq("")
        ]
        queue.extend((index, round_name) for index in pending)
    random.Random(20260502).shuffle(queue)
    logger.info("rows needing recovery: %d", len(queue))

    for done, (index, round_name) in enumerate(queue, start=1):
        row = frame.loc[index].to_dict()
        meta = lookup.get(str(row.get("question_identifier")), {})
        question = str(meta.get("question_text", ""))
        options = _options_for({**row, **meta})
        raw = str(row[text_column(round_name)])

        _assert_blind(cascade, cascade._build_user_prompt(question, options, raw),
                      row, question, options, raw)
        try:
            answer, method = cascade.extract_answer(question, options, raw)
        except Exception as exc:                      # keep the run alive
            logger.warning("judge failed on %s/%s: %s", index, round_name, exc)
            answer, method = ABSTAIN, "judge_error"
        frame.at[index, judged_column(round_name)] = (
            "" if answer == ABSTAIN else answer)
        frame.at[index, method_column(round_name)] = method

        if checkpoint is not None and done % flush_every == 0:
            frame.to_parquet(checkpoint, index=False)
            logger.info("checkpoint at %d/%d", done, len(queue))

    frame = add_effective_answers(frame, rounds)
    if checkpoint is not None:
        frame.to_parquet(checkpoint, index=False)
    return frame


def add_effective_answers(frame: pd.DataFrame,
                          rounds: Sequence[str] = ROUNDS) -> pd.DataFrame:
    """
    Regex answer where it exists, judge answer where it does not.

    The original column is never overwritten, so any reader can see which
    answers came from the extractor and which from a judge.
    """
    out = frame.copy()
    for round_name in rounds:
        regex = out[answer_column(round_name)].astype(str).str.strip()
        judged = (out[judged_column(round_name)].astype(str).str.strip()
                  if judged_column(round_name) in out.columns
                  else pd.Series("", index=out.index))
        out[effective_column(round_name)] = regex.where(regex.ne(""), judged)
    return out


def calibrate(
    trials: pd.DataFrame,
    cascade: Any,
    questions: Optional[pd.DataFrame] = None,
    sample_size: int = 150,
    seed: int = 20260502,
    round_name: str = "round1",
) -> Dict[str, Any]:
    """
    Agreement between judge and regex on rows the REGEX ALREADY READ.

    This is the number that licenses the recovered answers. Reported, never
    assumed: a judge that disagrees with the regex where the regex was
    confident is not a judge whose verdicts should enter the analysis.
    """
    _require_text(trials, [round_name])
    parsed = trials[
        trials[answer_column(round_name)].astype(str).str.strip().ne("")
        & trials[text_column(round_name)].astype(str).str.strip().ne("")
    ]
    if parsed.empty:
        return {"n": 0, "agreement": float("nan")}
    sample = parsed.sample(n=min(sample_size, len(parsed)), random_state=seed)
    lookup = ({str(r["question_identifier"]): r
               for r in questions.to_dict("records")}
              if questions is not None else {})

    agree = 0
    disagreements: List[Dict[str, str]] = []
    for row in sample.to_dict("records"):
        meta = lookup.get(str(row.get("question_identifier")), {})
        try:
            answer, _ = cascade.extract_answer(
                str(meta.get("question_text", "")),
                _options_for({**row, **meta}),
                str(row[text_column(round_name)]))
        except Exception:
            continue
        expected = str(row[answer_column(round_name)]).strip().upper()
        if str(answer).strip().upper() == expected:
            agree += 1
        elif len(disagreements) < 20:
            disagreements.append({"regex": expected, "judge": str(answer)})
    return {
        "n": int(len(sample)),
        "n_agree": agree,
        "agreement": agree / len(sample),
        "example_disagreements": disagreements,
    }


def recovery_summary(frame: pd.DataFrame,
                     rounds: Sequence[str] = ROUNDS) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for round_name in rounds:
        regex_blank = frame[answer_column(round_name)].astype(str).str.strip().eq("")
        effective_blank = (frame[effective_column(round_name)]
                           .astype(str).str.strip().eq(""))
        methods = (frame[method_column(round_name)].astype(str)
                   .replace("", pd.NA).dropna().value_counts().to_dict())
        summary[round_name] = {
            "unparsed_by_regex": int(regex_blank.sum()),
            "still_unresolved": int(effective_blank.sum()),
            "recovered": int(regex_blank.sum() - effective_blank.sum()),
            "methods": methods,
        }
    return summary


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
    parser.add_argument("--env", type=Path,
                        default=_ROOT.parent / ".env",
                        help="dotenv file holding the provider keys")
    args = parser.parse_args()

    # Keys live in Code/.env, which is gitignored. Loaded here rather than
    # required in the shell so the command is the same on every machine.
    if args.env.exists():
        import os
        for raw in args.env.read_text(encoding="utf-8",
                                      errors="ignore").splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            os.environ.setdefault(key.strip(),
                                  value.strip().strip('"').strip("'"))
        logger.info("loaded provider keys from %s", args.env)

    import yaml
    from src.agent_wrappers.judge_agent import JudgeCascade

    trials = pd.read_parquet(args.trials)
    try:
        _require_text(trials, ROUNDS)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    questions = (pd.read_parquet(args.questions)
                 if args.questions.exists() else None)
    cascade = JudgeCascade(yaml.safe_load(
        args.models_config.read_text(encoding="utf-8")))
    out = args.out or args.trials.with_name(args.trials.stem + "_judged.parquet")

    report: Dict[str, Any] = {}
    if args.calibrate:
        report["calibration"] = calibrate(
            trials, cascade, questions, sample_size=args.calibrate)
        logger.info("judge/regex agreement: %.4f on n=%d",
                    report["calibration"]["agreement"],
                    report["calibration"]["n"])

    recovered = recover(trials, cascade, questions, checkpoint=out)
    report["recovery"] = recovery_summary(recovered)
    report["judge_usage"] = cascade.usage_stats
    out.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", out)
    logger.info("summary: %s", json.dumps(report["recovery"], default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
