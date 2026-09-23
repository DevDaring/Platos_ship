"""
gsm_symbolic.py — X4 pool: a contamination probe with VALID gold answers.

Why this replaces the Phase-2 perturbation
------------------------------------------
`Code_Phase_2/CPU_Only/src/perturbed_gsm8k.py` multiplied every standalone
integer in the question by 4 and multiplied the gold answer by 4. That is only
correct when the answer is a linear function of every scaled operand. It is not
for percentages, fractions, or ratio quantities, and the pool contains all
three. Verified examples in the released pool:

  gsm8k_perturbed_0005: saved gold 10,400; true answer 41,600
  gsm8k_perturbed_0007: saved gold 1,540,000; true answer 7,840,000
  gsm8k_perturbed_0000: "4/8 an ounce" — the fraction itself was scaled

The saved probe agreed with the generated label on 11 of 97 items. Section 4.5
of the reviewed paper (solo accuracy "falling" 76.9% -> 21.6%) therefore
measures label corruption, not difficulty. It is quarantined, not deleted
(tools/quarantine_perturbed.py), and replaced by this.

GSM-Symbolic (Mirzadeh et al., ICLR 2025, arXiv:2410.05229) generates new
instances from 100 GSM8K templates with symbolic variables, so the answer is
computed by the template rather than guessed at. Each row carries `original_id`
into the GSM8K test split, which gives the matched original for free.

Claim this licenses: "the WR-minus-R gain on template-regenerated items is X
points against Y on the matched originals". NOT "contamination is ruled out" —
a new instance of a memorised template is still a memorised template.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .extraction import normalise_answer
from .seeding import derive_rng

logger = logging.getLogger("platos_ship3.gsm_symbolic")

_FINAL_RE = re.compile(r"####\s*(-?[\d,]+\.?\d*)")


def _parse_gsm_answer(answer_text: str) -> Optional[str]:
    """GSM8K/GSM-Symbolic answers end with '#### <number>'."""
    if not answer_text:
        return None
    match = _FINAL_RE.search(str(answer_text))
    if not match:
        return None
    return normalise_answer(match.group(1))


def _wrong_answer_pool(correct: str, question_id: str, master_seed: int,
                       n: int = 5) -> List[str]:
    """
    Plausible wrong targets for the anchored personas.

    Built from the correct value by small multiplicative and additive
    perturbations — near misses a model might actually produce. Deterministic,
    so the pool regenerates identically.
    """
    rng = derive_rng(master_seed, "wrongpool", question_id)
    try:
        value = float(correct)
    except (TypeError, ValueError):
        return [str(i) for i in range(1, n + 1)]

    candidates: List[str] = []
    for factor in (2.0, 0.5, 1.5, 10.0, 0.1):
        candidates.append(normalise_answer(value * factor))
    for delta in (1, -1, 2, -2, 10, -10):
        candidates.append(normalise_answer(value + delta))

    seen, pool = {normalise_answer(value)}, []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            pool.append(candidate)
    rng.shuffle(pool)
    return pool[:n]


def _load_symbolic_frame(hf_dataset: str, hf_config: str,
                         cache_dir: Optional[str]) -> "pd.DataFrame":
    """
    Load the dataset, preferring `datasets` and falling back to direct parquet.

    The fallback exists because `huggingface_hub`'s transport can fail on some
    stacks with a brotli decoding error before a single byte of data is used.
    That is an environment fault, not a data problem, and it should not be able
    to block an experiment — the datasets-server exposes the same parquet with
    identical columns, so read it directly and carry on.
    """
    try:
        from datasets import load_dataset

        logger.info("Loading %s (config=%s) via `datasets`…", hf_dataset, hf_config)
        return load_dataset(hf_dataset, hf_config, split="test",
                            cache_dir=cache_dir).to_pandas()
    except Exception as exc:
        logger.warning("`datasets` load failed (%s: %s); falling back to the "
                       "parquet endpoint.", type(exc).__name__, str(exc)[:120])

    import io
    import json as _json
    import os
    import urllib.request

    import pandas as _pd

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

    def _get(url: str) -> bytes:
        request = urllib.request.Request(
            url, headers={"Accept-Encoding": "identity"})   # avoids the brotli path
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.read()

    index = f"https://huggingface.co/api/datasets/{hf_dataset}/parquet/{hf_config}/test"
    shards = _json.loads(_get(index))
    if not shards:
        raise RuntimeError(f"No parquet shards listed for {hf_dataset}/{hf_config}")
    logger.info("Reading %d parquet shard(s) directly.", len(shards))
    return _pd.concat([_pd.read_parquet(io.BytesIO(_get(s))) for s in shards],
                      ignore_index=True)


def build_gsm_symbolic_pool(
    output_path: Path,
    master_seed: int,
    hf_dataset: str = "apple/GSM-Symbolic",
    hf_config: str = "main",
    instance_index: int = 0,
    n_questions: int = 100,
    pair_with_originals: bool = True,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build the X4 pool: one regenerated instance per template, plus (optionally)
    the matched original GSM8K item, in the same schema as question_pool.parquet.

    Licence note: GSM-Symbolic is CC-BY-NC-ND-4.0. Evaluation use is fine; the
    release ships item identifiers and per-trial logs, NOT redistributed or
    modified question text.
    """
    path = Path(output_path)
    if path.exists():
        logger.info("GSM-Symbolic pool exists at %s — loading.", path)
        return pd.read_parquet(path)

    frame = _load_symbolic_frame(hf_dataset, hf_config, cache_dir)

    required = {"id", "instance", "question", "answer", "original_id"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{hf_dataset} is missing expected columns: {missing}")

    selected = (
        frame[frame["instance"] == instance_index]
        .sort_values("id")
        .head(int(n_questions))
        .copy()
    )
    logger.info("Selected %d regenerated instances (instance=%d).",
                len(selected), instance_index)

    rows: List[Dict[str, Any]] = []
    dropped = 0
    for record in selected.to_dict("records"):
        correct = _parse_gsm_answer(record["answer"])
        if correct is None:
            dropped += 1
            continue
        question_id = f"gsmsym_{int(record['id']):04d}"
        rows.append(
            {
                "question_identifier": question_id,
                "source_dataset": "gsm_symbolic",
                "subject_category": "math",
                "question_text": record["question"],
                "answer_options": None,
                "correct_answer": correct,
                "correct_answer_full_text": correct,
                "wrong_answer_pool": json.dumps(
                    _wrong_answer_pool(correct, question_id, master_seed)),
                "difficulty_stratum": "gsm_symbolic",
                "random_seed_used": master_seed,
                "included_in_mitigation_subset": False,
                "gsm_symbolic_id": int(record["id"]),
                "gsm_symbolic_instance": int(record["instance"]),
                "gsm8k_original_id": int(record["original_id"]),
                "pair_role": "regenerated",
            }
        )

        if pair_with_originals:
            original_correct = _parse_gsm_answer(record.get("original_answer", ""))
            original_question = record.get("original_question")
            if original_correct and original_question:
                original_id = f"gsmorig_{int(record['original_id']):04d}"
                rows.append(
                    {
                        "question_identifier": original_id,
                        "source_dataset": "gsm8k_original",
                        "subject_category": "math",
                        "question_text": original_question,
                        "answer_options": None,
                        "correct_answer": original_correct,
                        "correct_answer_full_text": original_correct,
                        "wrong_answer_pool": json.dumps(
                            _wrong_answer_pool(original_correct, original_id,
                                               master_seed)),
                        "difficulty_stratum": "gsm8k_original",
                        "random_seed_used": master_seed,
                        "included_in_mitigation_subset": False,
                        "gsm_symbolic_id": int(record["id"]),
                        "gsm_symbolic_instance": int(record["instance"]),
                        "gsm8k_original_id": int(record["original_id"]),
                        "pair_role": "original",
                    }
                )

    pool = pd.DataFrame(rows).drop_duplicates(subset=["question_identifier"])
    if dropped:
        logger.warning("Dropped %d items whose '#### answer' did not parse.", dropped)

    path.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(path, index=False)
    logger.info(
        "GSM-Symbolic pool: %d items (%d regenerated, %d matched originals) -> %s",
        len(pool),
        int((pool["pair_role"] == "regenerated").sum()),
        int((pool["pair_role"] == "original").sum()),
        path,
    )
    return pool


def audit_pool_answers(pool: pd.DataFrame, sample_size: int = 20,
                       output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Emit a hand-audit sheet for the gold answers.

    The Phase-2 failure was trusting a generator's arithmetic. Even with
    template-computed answers, a fixed sample is checked by hand and the result
    stated in the appendix — cheap insurance against repeating the mistake.
    """
    take = min(int(sample_size), len(pool))
    if take == 0:
        return pd.DataFrame()
    sample = pool.sample(n=take, random_state=20260502).copy()
    sheet = sample[["question_identifier", "pair_role", "question_text",
                    "correct_answer"]].copy()
    sheet["auditor_recomputed_answer"] = ""
    sheet["auditor_agrees"] = ""
    sheet["auditor_notes"] = ""
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.to_csv(path, index=False)
        logger.info("Gold-answer audit sheet (%d items) -> %s", len(sheet), path)
    return sheet
