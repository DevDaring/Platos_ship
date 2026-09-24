#!/usr/bin/env python3
"""
run_big_probe.py — the output-distribution probe on Llama-3.1-70B, bf16.

Turns a stated limitation into a result. The reviewed paper probes only
Llama-3.1-8B and says so:

    "we cannot say whether the strong models that resist wrong peers do so by
     feeling less pull or by overriding more of it. Separating those two
     accounts would need the same read-out on a strong open-weight model."

Running the same read-out on a 70B answers it:

  * small distributional movement  -> the strong model feels less pull;
  * movement comparable to the 8B but no flip -> it overrides more.

Either way the paper gains a mechanism sentence it cannot currently write.

Measurement uses GPU_Only/src/corrected_probe.py, not the Phase-2 probe:
exact tokenised candidate scoring (the old one matched on first character,
so numeric answers sharing a leading digit collided), off-candidate mass
retained rather than normalised away, and one target per (question, replicate)
fixed across every condition so the difference-in-differences means something.

Checkpointed per stage: an interrupted run resumes without re-paying for GPU
time already spent.

    python3 run_big_probe.py --tensor-parallel-size 2
    python3 run_big_probe.py --dry-run          # 2 questions, verifies wiring
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
PHASE3_ROOT = HERE.parent
for path in (str(PHASE3_ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import pandas as pd  # noqa: E402

from GPU_Only.src.corrected_probe import (  # noqa: E402
    ExactCandidateScorer,
    build_probe_targets,
    difference_in_differences,
    difference_in_differences_by_task,
)
from src.contexts import PeerMessage, build_revision_prompt, build_round0_prompt  # noqa: E402
from src.extraction import extract_answer_regex, normalise_answer  # noqa: E402
from src.seeding import derive_seed  # noqa: E402

logger = logging.getLogger("platos_ship3.big_probe")

# The three conditions the probe needs: the matched baseline, the homogeneous
# control (which supplies the difference-in-differences reference), and the
# treatment.
PROBE_CONDITIONS = ["R", "E", "WR"]


def load_inputs(questions_path: Path, personas_path: Path,
                mcq_only: bool) -> Tuple[pd.DataFrame, Dict[str, List[dict]]]:
    questions = pd.read_parquet(questions_path)
    if mcq_only:
        # First-position scoring is exact for single-token option letters.
        # Numeric answers need full-sequence teacher forcing, which is slower
        # and is opt-in via --include-numeric.
        before = len(questions)
        questions = questions[questions["answer_options"].notna()]
        logger.info("Restricted to %d multiple-choice items (from %d).",
                    len(questions), before)

    personas = pd.read_parquet(personas_path)
    index: Dict[str, List[dict]] = {}
    for row in personas.to_dict("records"):
        index.setdefault(row["question_identifier"], []).append(row)
    return questions.reset_index(drop=True), index


def candidates_for(question: dict) -> List[str]:
    """Answer options for an MCQ item, or a numeric candidate set."""
    options = question.get("answer_options")
    if options is not None and isinstance(options, (str, list)):
        if isinstance(options, str):
            options = json.loads(options)
        if options:
            return [chr(65 + i) for i in range(len(options))]
    pool = question.get("wrong_answer_pool")
    if isinstance(pool, str):
        pool = json.loads(pool)
    return [normalise_answer(question["correct_answer"])] + [
        normalise_answer(x) for x in (pool or [])
    ]


def build_peers_for(condition: str, question: dict, target: str,
                    personas: List[dict], self_texts: List[str]) -> List[PeerMessage]:
    if condition == "R":
        return []
    if condition == "E":
        return [
            PeerMessage(display_name=f"Agent_{i}", text=text,
                        final_answer=extract_answer_regex(text),
                        peer_source="self_samples", anchor_mode="self")
            for i, text in enumerate(self_texts[:2])
        ]
    chosen = [p for p in personas
              if str(p.get("assigned_wrong_answer_letter_or_value")) == str(target)]
    chosen = (chosen or personas)[:2]
    while len(chosen) < 2 and personas:
        chosen.append(personas[len(chosen) % len(personas)])
    return [
        PeerMessage(
            display_name=f"Agent_{i}",
            text=p.get("generated_persona_text", ""),
            final_answer=p.get("assigned_wrong_answer_letter_or_value"),
            message_generator_model=p.get("generator_model_name", ""),
            peer_source="anchored_wrong", anchor_mode="wrong",
            assigned_target=p.get("assigned_wrong_answer_letter_or_value"),
        )
        for i, p in enumerate(chosen)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-70B-Instruct")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--questions", type=int, default=300)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--include-numeric", action="store_true",
                        help="also score GSM8K items (full-sequence scoring)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inputs", default=str(HERE / "inputs"))
    parser.add_argument("--out", default=str(HERE / "results"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(HERE / "big_probe.log", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )

    inputs_dir, out_dir = Path(args.inputs), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "big_probe_trials.parquet"

    by_id: Dict[str, dict] = {}
    questions, personas = load_inputs(
        inputs_dir / "question_pool.parquet",
        inputs_dir / "dumb_personas.parquet",
        mcq_only=not args.include_numeric,
    )
    if args.dry_run:
        questions = questions.head(2)
        args.replicates = 1
    else:
        questions = questions.head(args.questions)

    by_id.update({q["question_identifier"]: q
                  for q in questions.to_dict("records")})
    targets = build_probe_targets(questions.to_dict("records"), personas,
                                  args.replicates, args.seed)
    logger.info("%d questions x %d replicates; %d fixed targets.",
                len(questions), args.replicates, len(targets))

    # ── resume ────────────────────────────────────────────────────────────
    done = set()
    if trials_path.exists():
        existing = pd.read_parquet(trials_path)
        done = set(zip(existing["question_identifier"], existing["replicate"],
                       existing["condition"]))
        logger.info("Resuming: %d trials already complete.", len(done))

    # ── model ─────────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    logger.info("Loading %s (bf16, tp=%d)…", args.model, args.tensor_parallel_size)
    started = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",                       # never quantise: see README
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    scorer = ExactCandidateScorer(tokenizer)
    logger.info("Model ready in %.1f min.", (time.time() - started) / 60)

    def _system_role_supported() -> bool:
        """
        Not every chat template accepts a system role.

        Gemma's raises outright, so hardcoding the two-message form would
        have failed on gemma-3-4b and gemma-3-27b AFTER the checkpoint was
        downloaded and the GPUs were billing. Probed once here rather than
        assumed, and recorded in the run metadata so a reader can see which
        prompt shape each model actually received.
        """
        try:
            tokenizer.apply_chat_template(
                [{"role": "system", "content": "probe"},
                 {"role": "user", "content": "probe"}],
                tokenize=False, add_generation_prompt=True,
            )
            return True
        except Exception as exc:                       # template rejects it
            logger.info("chat template rejects a system role (%s); folding "
                        "the system text into the user turn", type(exc).__name__)
            return False

    system_role_supported = _system_role_supported()

    def chat(system: str, user: str) -> str:
        if system_role_supported:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": user}]
        else:
            # Same text, one turn. The instruction still precedes the
            # question, so the model sees identical content; only the
            # envelope differs, and it differs identically in every
            # condition, which is what the matched design requires.
            messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def _teacher_force(prefixes: List[str],
                       candidate_sets: List[List[str]]) -> List[Dict[str, float]]:
        """
        Exact P(candidate | prompt) for candidates that span several tokens.

        A numeric answer like "150" is three tokens, so it cannot be read off
        the first-position table; the released probe matched such answers by
        their FIRST CHARACTER, which silently merges "1", "12" and "150".
        Here each multi-token candidate is appended to the prompt and scored by
        summing the log-probabilities of its own tokens under teacher forcing.

        All (prompt, candidate) pairs across the whole batch go in one vLLM
        call, so the extra cost is throughput, not round trips.
        """
        jobs, sequences = [], []
        for index, (prefix, candidates) in enumerate(zip(prefixes, candidate_sets)):
            for candidate in scorer.needs_teacher_forcing(candidates):
                # Only the WIDTH is taken from the local tokenisation; the
                # positions come from vLLM's own prompt_token_ids, because vLLM
                # prepends BOS and its indices are shifted by one.
                start, end = scorer.continuation_span(prefix, candidate)
                jobs.append((index, candidate, end - start))
                sequences.append(f"{prefix} {candidate}")

        results: List[Dict[str, float]] = [{} for _ in prefixes]
        if not sequences:
            return results

        # max_tokens=1 because nothing is generated: the scores come from
        # prompt_logprobs over the forced continuation.
        params = SamplingParams(temperature=0.0, max_tokens=1,
                                prompt_logprobs=0, seed=args.seed)
        outputs = llm.generate(sequences, params, use_tqdm=True)
        for (index, candidate, width), output in zip(jobs, outputs):
            results[index][candidate] = scorer.score_from_prompt_logprobs(
                output.prompt_logprobs, output.prompt_token_ids, width)
        return results

    def distribution(prompts: List[str], candidate_sets: List[List[str]]):
        """
        Exact candidate probabilities.

        Single-token candidates (multiple-choice letters A-J) come from the
        first-position table in one pass. Multi-token candidates (GSM8K
        numbers) are then scored by teacher forcing. Probabilities are
        absolute and never renormalised, so the residual mass the model put
        outside the candidate set stays reportable.
        """
        prefixes = [p + " Final answer:" for p in prompts]
        params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20,
                                seed=args.seed)
        outputs = llm.generate(prefixes, params, use_tqdm=True)

        staged = []
        any_forced = False
        for output, candidates in zip(outputs, candidate_sets):
            table = output.outputs[0].logprobs
            first = ({tid: lp.logprob for tid, lp in table[0].items()}
                     if table else {})
            scores, outside = scorer.score_from_first_position(first, candidates)
            staged.append((scores, outside, candidates))
            if scorer.needs_teacher_forcing(candidates):
                any_forced = True

        forced = (_teacher_force(prefixes, candidate_sets) if any_forced
                  else [{} for _ in prefixes])

        results = []
        for (scores, outside, _), forced_scores in zip(staged, forced):
            if forced_scores:
                scores, outside = scorer.merge_teacher_forced(scores, forced_scores)
            results.append((
                {s.candidate: (s.probability if s.probability is not None else 0.0)
                 for s in scores},
                outside,
                [s.candidate for s in scores if s.collides_with],
                # The methods ACTUALLY used, read off the scores. Deriving this
                # from the question instead would mislabel every GSM8K row,
                # because their `answer_options` cell holds NaN, not None.
                sorted({s.scoring_method for s in scores}),
            ))
        return results

    def generate(prompts: List[str], seeds: List[int]) -> List[str]:
        """
        One batched call with a per-prompt seed.

        Issuing these one at a time would leave most of a 70B's throughput
        idle; vLLM accepts a SamplingParams list aligned with the prompts, so
        the decode seeds stay per-trial and logged while the batch stays whole.
        """
        params = [SamplingParams(temperature=0.7, max_tokens=600, seed=s)
                  for s in seeds]
        outputs = llm.generate(prompts, params, use_tqdm=True)
        return [o.outputs[0].text for o in outputs]

    # ── stage A: cached Round-0, shared by every condition ────────────────
    logger.info("Stage A: Round-0 cache")
    r0: Dict[Tuple[str, int], Dict[str, Any]] = {}
    rows = questions.to_dict("records")
    a_prompts, a_keys, a_cands = [], [], []
    for question in rows:
        for replicate in range(args.replicates):
            system, user = build_round0_prompt(question["question_text"],
                                               question.get("answer_options"))
            a_prompts.append(chat(system, user))
            a_keys.append((question["question_identifier"], replicate))
            a_cands.append(candidates_for(question))

    a_dists = distribution(a_prompts, a_cands)
    a_texts = generate(a_prompts,
                       [derive_seed(args.seed, "r0", q, r) for q, r in a_keys])
    for key, text, (dist, outside, collisions, methods) in zip(
            a_keys, a_texts, a_dists):
        r0[key] = {"text": text, "answer": extract_answer_regex(text),
                   "dist": dist, "outside": outside, "collisions": collisions,
                   "methods": methods}

    # ── stage B: one revision per condition ───────────────────────────────
    collected: List[Dict[str, Any]] = []
    for condition in PROBE_CONDITIONS:
        logger.info("Stage B: condition %s", condition)
        prompts, keys, cands = [], [], []
        for question in rows:
            qid = question["question_identifier"]
            for replicate in range(args.replicates):
                if (qid, replicate, condition) in done:
                    continue
                target = targets.get((qid, replicate))
                if target is None:
                    continue
                cached = r0[(qid, replicate)]
                self_texts = [r0[(qid, (replicate + k) % args.replicates)]["text"]
                              for k in (1, 2)]
                peers = build_peers_for(condition, question, target,
                                        personas.get(qid, []), self_texts)
                system, user = build_revision_prompt(
                    question["question_text"], question.get("answer_options"),
                    cached["text"], peers,
                    {"R": "none", "E": "self_samples", "WR": "anchored_wrong"}[condition],
                )
                prompts.append(chat(system, user))
                keys.append((qid, replicate, target))
                cands.append(candidates_for(question))

        if not prompts:
            continue
        dists = distribution(prompts, cands)
        texts = generate(prompts, [derive_seed(args.seed, "r1", condition, q, r)
                                   for q, r, _ in keys])

        for (qid, replicate, target), text, (dist, outside, collisions,
                                            methods) in zip(keys, texts, dists):
            question = by_id[qid]
            cached = r0[(qid, replicate)]
            correct = normalise_answer(question["correct_answer"])
            target_n = normalise_answer(target)
            r0_answer = cached["answer"]
            r1_answer = extract_answer_regex(text)
            collected.append({
                "question_identifier": qid,
                "source_dataset": question.get("source_dataset"),
                "replicate": replicate,
                "condition": condition,
                "focal_model": args.model,
                "fixed_target": target_n,
                "correct_answer": correct,
                "round0_answer": normalise_answer(r0_answer),
                "round1_answer": normalise_answer(r1_answer),
                # The RAW generations are kept. The regex is lossy -- on this
                # model ~19% of Round-1 generations do not yield an answer --
                # and discarding the text makes that loss unrecoverable
                # without re-renting the GPU. Keeping it lets a judge
                # cascade re-read the failures offline, and lets any reader
                # audit the extractor against the text it was run on.
                "round0_text": cached["text"],
                "round1_text": text,
                "prob_mass_round0_on_target": cached["dist"].get(target_n),
                "prob_mass_round1_on_target": dist.get(target_n),
                "delta_prob_mass_toward_target":
                    (dist.get(target_n) or 0.0) - (cached["dist"].get(target_n) or 0.0),
                "prob_mass_round0_on_correct": cached["dist"].get(correct),
                "prob_mass_round1_on_correct": dist.get(correct),
                "delta_prob_mass_toward_correct":
                    (dist.get(correct) or 0.0) - (cached["dist"].get(correct) or 0.0),
                # Kept, not normalised away — the released probe discarded this.
                "mass_outside_candidates_round0": cached["outside"],
                "mass_outside_candidates_round1": outside,
                "candidate_collisions": ",".join(collisions),
                # What was actually used, for round 1 and for the cached
                # round 0, so a mislabelled row cannot reach the analysis.
                "scoring_path": ",".join(methods),
                "scoring_path_round0": ",".join(cached.get("methods", [])),
                "round0_was_correct": normalise_answer(r0_answer) == correct,
                "round1_was_correct": normalise_answer(r1_answer) == correct,
            })

        # Flush after every condition: GPU time already spent is never re-paid.
        frame = pd.DataFrame(collected)
        if trials_path.exists():
            frame = pd.concat([pd.read_parquet(trials_path), frame],
                              ignore_index=True).drop_duplicates(
                subset=["question_identifier", "replicate", "condition"], keep="last")
        frame.to_parquet(trials_path, index=False)
        collected = []
        logger.info("  %s written (%d rows total).", condition, len(frame))

    # ── contrast ──────────────────────────────────────────────────────────
    trials = pd.read_parquet(trials_path)
    # Per task family: multiple choice and GSM8K are on different probability
    # scales, so a pooled mean would be dominated by multiple choice.
    contrasts = difference_in_differences_by_task(trials, "WR", "E")
    pd.DataFrame(contrasts).to_parquet(
        out_dir / "big_probe_contrast.parquet", index=False)
    contrast = next((c for c in contrasts if c.get("report")), contrasts[-1])

    audit = []
    for question in rows:
        candidates = candidates_for(question)
        single = scorer.single_token_candidates(candidates)
        audit.append({
            "question_identifier": question["question_identifier"],
            "n_candidates": len(candidates),
            "n_single_token": len(single),
            "collisions": json.dumps(scorer.collision_report(candidates)),
        })
    pd.DataFrame(audit).to_parquet(
        out_dir / "big_probe_candidates.parquet", index=False)

    import torch
    with open(out_dir / "big_probe_meta.json", "w", encoding="utf-8") as handle:
        json.dump({
            "model": args.model,
            "dtype": "bfloat16",
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_len,
            "gpu": torch.cuda.get_device_name(0),
            "n_gpus": torch.cuda.device_count(),
            "seed": args.seed,
            "questions": len(rows),
            "replicates": args.replicates,
            "mcq_only": not args.include_numeric,
            # Which prompt envelope this model received. Gemma's template
            # rejects a system role, so its system text is folded into the
            # user turn; every other model keeps the two-message form.
            "system_role_supported": system_role_supported,
            "wall_clock_minutes": round((time.time() - started) / 60, 1),
            "contrasts_by_task": contrasts,
        }, handle, indent=2, default=str)

    for row in contrasts:
        label = row.get("source_dataset", "?")
        flag = "" if row.get("report") else "   (pooled - do not report)"
        logger.info("WR-minus-E [%s]: %.6f (95%% CI [%.6f, %.6f], n=%d)%s",
                    label, row.get("estimate", float("nan")),
                    row.get("ci_low", float("nan")),
                    row.get("ci_high", float("nan")),
                    row.get("n_pairs", 0), flag)
    logger.info("Results in %s — pull them, then DESTROY the instance.", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
