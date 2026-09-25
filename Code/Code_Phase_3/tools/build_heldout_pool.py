#!/usr/bin/env python3
"""
build_heldout_pool.py — Experiment B's held-out question pool (PREREG_PHASE4.md).

420 MMLU-Pro test questions, 60 from each of seven subjects that need little or
no arithmetic (biology, health, history, law, other, philosophy, psychology),
none of which appears in the main 300-question pool (matched on question text).
Sampled once with a fixed seed; the pool is written in the same format as
results/processed/question_pool.parquet.

    python tools/build_heldout_pool.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results/phase4/B/heldout_pool.parquet"
SUBJECTS = ["biology", "health", "history", "law", "other", "philosophy", "psychology"]
PER_SUBJECT = 60
SEED = 20260925


def main() -> int:
    import run_all
    from datasets import load_dataset

    run_all.load_env(ROOT)
    main_pool = pd.read_parquet(ROOT / "results/processed/question_pool.parquet")
    seen = {" ".join(str(t).split()) for t in main_pool["question_text"]}

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test").to_pandas()
    rng = random.Random(SEED)
    rows = []
    for subject in SUBJECTS:
        sub = ds[ds["category"].str.lower() == subject]
        sub = sub[~sub["question"].map(lambda t: " ".join(str(t).split()) in seen)]
        ids = sorted(sub["question_id"].tolist())
        chosen = set(rng.sample(ids, PER_SUBJECT))
        for _, r in sub[sub["question_id"].isin(chosen)].sort_values("question_id").iterrows():
            options = [str(o) for o in r["options"]]
            letter = str(r["answer"]).strip()
            index = ord(letter) - 65
            rows.append({
                "question_identifier": f"mmluho_{len(rows):04d}",
                "source_dataset": "mmlu_pro",
                "subject_category": subject,
                "question_text": str(r["question"]),
                "answer_options": json.dumps(options),
                "correct_answer": letter,
                "correct_answer_full_text": options[index],
                "wrong_answer_pool": json.dumps([chr(65 + i) for i in range(len(options))
                                                 if i != index]),
                "difficulty_stratum": "heldout",
                "random_seed_used": SEED,
                "included_in_mitigation_subset": False,
                "mmlu_pro_question_id": int(r["question_id"]),
            })
    pool = pd.DataFrame(rows)
    assert pool["question_text"].map(lambda t: " ".join(t.split()) in seen).sum() == 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(OUT, index=False)
    print(f"held-out pool: {len(pool)} questions -> {OUT}")
    print(pool["subject_category"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
