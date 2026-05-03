"""
dataset_builder.py — Downloads, samples, and prepares the question pool.

Handles MMLU-Pro (10 subjects × 20 questions) and GSM8K (100 questions)
with difficulty stratification using Llama 3.1 8B as a probe model.
"""

import os
import re
import json
import random
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

logger = logging.getLogger("platos_ship.dataset_builder")

# MMLU-Pro subjects in priority order
MMLU_PRO_SUBJECTS = [
    "math", "physics", "chemistry", "biology", "computer science",
    "economics", "history", "law", "philosophy", "psychology",
]


def load_mmlu_pro(cache_dir: Path) -> pd.DataFrame:
    """Download and return MMLU-Pro test split."""
    from datasets import load_dataset
    logger.info("Loading MMLU-Pro dataset...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", cache_dir=str(cache_dir))
    df = ds.to_pandas()
    logger.info(f"MMLU-Pro loaded: {len(df)} rows, columns: {list(df.columns)}")
    return df


def load_gsm8k(cache_dir: Path) -> pd.DataFrame:
    """Download and return GSM8K test split."""
    from datasets import load_dataset
    logger.info("Loading GSM8K dataset...")
    # Use canonical HF name ("openai/gsm8k"); the old alias "gsm8k" is deprecated
    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=str(cache_dir))
    df = ds.to_pandas()
    logger.info(f"GSM8K loaded: {len(df)} rows, columns: {list(df.columns)}")
    return df


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract numeric answer from GSM8K answer field (after ####)."""
    match = re.search(r"####\s*(.+)$", answer_text.strip())
    if match:
        return match.group(1).strip().replace(",", "")
    # Fallback: last number in the text
    numbers = re.findall(r"-?[\d,]+\.?\d*", answer_text)
    if numbers:
        return numbers[-1].replace(",", "")
    return answer_text.strip()


def generate_gsm8k_wrong_answers(correct_answer_str: str) -> List[str]:
    """Generate plausible wrong answers for GSM8K questions."""
    try:
        correct = float(correct_answer_str.replace(",", ""))
    except ValueError:
        return ["0", "1", "2", "3", "4"]

    wrongs = set()
    candidates = [
        correct + 1, correct - 1, correct * 2,
        correct / 2, correct + 10,
    ]
    for c in candidates:
        # Use integer formatting only when the value is a whole number
        try:
            c_int = int(round(c))
            c_str = str(c_int) if abs(c - c_int) < 1e-9 else f"{c:.2f}"
        except (OverflowError, ValueError):
            c_str = f"{c:.2f}"
        if c_str != correct_answer_str and c_str not in wrongs:
            wrongs.add(c_str)

    return list(wrongs)[:5]


def difficulty_probe_batch(
    questions: List[Dict],
    probe_agent,
    is_multiple_choice: bool = True,
) -> List[str]:
    """
    Run difficulty probe on a batch of questions using a local model.
    Returns list of 'probe_correct' or 'probe_incorrect' per question.
    """
    results = []
    for q in questions:
        if is_multiple_choice:
            options_text = "\n".join(
                f"{chr(65+i)}. {opt}" for i, opt in enumerate(q.get("options", []))
            )
            prompt = (
                f"Answer the following multiple choice question with just the letter.\n\n"
                f"Question: {q['question']}\n{options_text}\n\n"
                f"Final answer:"
            )
        else:
            prompt = (
                f"Solve the following math problem. Give only the numeric answer.\n\n"
                f"Question: {q['question']}\n\nFinal answer:"
            )

        try:
            response = probe_agent.generate_response(
                system_prompt="You are a helpful assistant. Answer concisely.",
                user_prompt=prompt,
                temperature=0.0,
                maximum_output_tokens=50,
            )
            raw = response.raw_text_output.strip()

            # Extract answer from probe response
            if is_multiple_choice:
                match = re.search(r"([A-J])", raw)
                probe_ans = match.group(1) if match else ""
                correct = str(q.get("correct_answer", "")).strip()
                results.append("probe_correct" if probe_ans == correct else "probe_incorrect")
            else:
                numbers = re.findall(r"-?[\d,]+\.?\d*", raw)
                probe_ans = numbers[-1].replace(",", "") if numbers else ""
                correct = str(q.get("correct_answer", "")).strip().replace(",", "")
                results.append("probe_correct" if probe_ans == correct else "probe_incorrect")
        except Exception as e:
            logger.warning(f"Probe failed for question: {e}")
            results.append("probe_incorrect")

    return results


def sample_mmlu_pro(
    df: pd.DataFrame,
    subjects: List[str],
    questions_per_subject: int,
    seed: int,
    probe_agent=None,
) -> List[Dict]:
    """Stratified sampling from MMLU-Pro with difficulty probe."""
    rng = random.Random(seed)
    all_questions = []

    # Map category column
    category_col = "category" if "category" in df.columns else "subject"

    for subject in subjects:
        subject_df = df[df[category_col].str.lower() == subject.lower()]
        if len(subject_df) == 0:
            raise ValueError(
                f"Subject '{subject}' not found in MMLU-Pro dataset. "
                f"Available categories: {sorted(df[category_col].unique().tolist())}"
            )
        if len(subject_df) < questions_per_subject:
            logger.warning(
                f"Subject '{subject}' has only {len(subject_df)} items "
                f"(need {questions_per_subject}). Using all available."
            )

        # Convert to list of dicts for probing
        candidates = []
        for idx, row in subject_df.iterrows():
            options = row.get("options", [])
            if isinstance(options, str):
                options = json.loads(options)
            elif hasattr(options, "tolist"):  # numpy array / pandas array
                options = options.tolist()
            elif options is None or (not isinstance(options, list) and not hasattr(options, "__iter__")):
                options = []
            else:
                options = list(options)  # ensure plain Python list

            # Determine correct answer letter
            answer = str(row.get("answer", ""))
            answer_idx = row.get("answer_index", None)

            if len(answer) == 1 and answer.isalpha():
                correct_letter = answer.upper()
            elif answer_idx is not None:
                correct_letter = chr(65 + int(answer_idx))
            else:
                correct_letter = answer

            correct_full = ""
            if options and correct_letter.isalpha():
                letter_idx = ord(correct_letter) - 65
                if 0 <= letter_idx < len(options):
                    correct_full = options[letter_idx]

            wrong_pool = []
            for i, opt in enumerate(options):
                letter = chr(65 + i)
                if letter != correct_letter:
                    wrong_pool.append(letter)

            candidates.append({
                "question": row["question"],
                "options": options,
                "correct_answer": correct_letter,
                "correct_answer_full_text": correct_full,
                "wrong_answer_pool": wrong_pool,
                "subject": subject,
                "original_index": idx,
            })

        # Run difficulty probe if agent available
        if probe_agent and candidates:
            strata = difficulty_probe_batch(candidates, probe_agent, is_multiple_choice=True)
            for c, s in zip(candidates, strata):
                c["difficulty_stratum"] = s
        else:
            # Without probe, assign random strata
            for c in candidates:
                c["difficulty_stratum"] = rng.choice(["probe_correct", "probe_incorrect"])

        # Stratified selection: aim for 50/50
        correct_pool = [c for c in candidates if c["difficulty_stratum"] == "probe_correct"]
        incorrect_pool = [c for c in candidates if c["difficulty_stratum"] == "probe_incorrect"]

        half = questions_per_subject // 2
        rng.shuffle(correct_pool)
        rng.shuffle(incorrect_pool)

        selected = correct_pool[:half] + incorrect_pool[:half]

        # Fill remainder if one stratum is short
        remaining = questions_per_subject - len(selected)
        if remaining > 0:
            leftovers = [c for c in candidates if c not in selected]
            rng.shuffle(leftovers)
            selected.extend(leftovers[:remaining])

        selected = selected[:questions_per_subject]
        all_questions.extend(selected)

    return all_questions


def sample_gsm8k(
    df: pd.DataFrame,
    count: int,
    seed: int,
    probe_agent=None,
) -> List[Dict]:
    """Sample GSM8K questions with difficulty probe."""
    rng = random.Random(seed)

    indices = list(range(len(df)))
    rng.shuffle(indices)

    candidates = []
    for idx in indices[:count * 3]:  # Over-sample for stratification
        row = df.iloc[idx]
        correct_str = extract_gsm8k_answer(row["answer"])
        wrong_pool = generate_gsm8k_wrong_answers(correct_str)

        candidates.append({
            "question": row["question"],
            "options": None,
            "correct_answer": correct_str,
            "correct_answer_full_text": correct_str,
            "wrong_answer_pool": wrong_pool,
            "subject": "mathematics_word_problem",
            "original_index": idx,
        })

    # Run probe
    if probe_agent and candidates:
        strata = difficulty_probe_batch(candidates, probe_agent, is_multiple_choice=False)
        for c, s in zip(candidates, strata):
            c["difficulty_stratum"] = s
    else:
        for c in candidates:
            c["difficulty_stratum"] = rng.choice(["probe_correct", "probe_incorrect"])

    # Stratified: aim for 50/50, accept 60/40
    correct_pool = [c for c in candidates if c["difficulty_stratum"] == "probe_correct"]
    incorrect_pool = [c for c in candidates if c["difficulty_stratum"] == "probe_incorrect"]

    half = count // 2
    rng.shuffle(correct_pool)
    rng.shuffle(incorrect_pool)

    selected = correct_pool[:half] + incorrect_pool[:half]
    remaining = count - len(selected)
    if remaining > 0:
        leftovers = [c for c in candidates if c not in selected]
        rng.shuffle(leftovers)
        selected.extend(leftovers[:remaining])

    return selected[:count]


def select_mitigation_subset(
    questions: List[Dict], seed: int, total: int = 100,
    mmlu_count: int = 67, gsm_count: int = 33,
) -> List[Dict]:
    """Select stratified subset for C5 mitigation condition."""
    rng = random.Random(seed)

    mmlu_qs = [q for q in questions if q.get("source_dataset") == "mmlu_pro"]
    gsm_qs = [q for q in questions if q.get("source_dataset") == "gsm8k"]

    rng.shuffle(mmlu_qs)
    rng.shuffle(gsm_qs)

    subset_ids = set()
    for q in mmlu_qs[:mmlu_count]:
        subset_ids.add(q["question_identifier"])
    for q in gsm_qs[:gsm_count]:
        subset_ids.add(q["question_identifier"])

    # Mark all questions
    for q in questions:
        q["included_in_mitigation_subset"] = q["question_identifier"] in subset_ids

    return questions


def build_question_pool(
    project_root: Path,
    probe_agent=None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Build the complete question pool and save to parquet.

    If the question pool parquet already exists (and this is not a dry-run),
    it is loaded directly without re-downloading datasets or re-running the
    difficulty probe.  This makes crash-resume free and reproducible.

    Args:
        project_root: Root directory of the project.
        probe_agent: Agent used for difficulty probing (optional).
        dry_run: If True, use minimal sample sizes.
    """
    with open(project_root / "config" / "experiment.yaml") as f:
        exp_config = yaml.safe_load(f)
    with open(project_root / "config" / "paths.yaml") as f:
        paths = yaml.safe_load(f)

    # ── Cache check: load parquet if it already exists and we are NOT in dry-run ──
    if not dry_run:
        qp_path = paths["question_pool_file"]
        if not Path(qp_path).is_absolute():
            qp_path = project_root / qp_path
        if Path(qp_path).exists():
            logger.info(f"Question pool already exists at {qp_path} — loading from cache")
            df = pd.read_parquet(str(qp_path))
            logger.info(f"Question pool loaded: {len(df)} questions")
            return df

    seed = exp_config["random_seed"]
    raw_dir = Path(os.path.expanduser(paths["raw_data_directory"]))
    if not raw_dir.is_absolute():
        raw_dir = project_root / raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        mmlu_per_subject = 1
        gsm_count = 1
        subjects = MMLU_PRO_SUBJECTS[:1]
    else:
        mmlu_per_subject = exp_config["questions_per_subject"]
        gsm_count = exp_config["gsm8k_question_count"]
        subjects = MMLU_PRO_SUBJECTS[:exp_config["mmlu_pro_subject_count"]]

    # Load datasets
    mmlu_df = load_mmlu_pro(raw_dir)
    gsm_df = load_gsm8k(raw_dir)

    logger.info(f"MMLU-Pro total rows: {len(mmlu_df)}")
    logger.info(f"GSM8K total rows: {len(gsm_df)}")

    # Sample
    mmlu_questions = sample_mmlu_pro(mmlu_df, subjects, mmlu_per_subject, seed, probe_agent)
    gsm_questions = sample_gsm8k(gsm_df, gsm_count, seed, probe_agent)

    # Build unified question pool
    all_questions = []
    for i, q in enumerate(mmlu_questions):
        q["question_identifier"] = f"mmlupro_{i:04d}"
        q["source_dataset"] = "mmlu_pro"
        q["subject_category"] = q["subject"]
        q["question_text"] = q["question"]
        q["answer_options"] = q["options"]
        q["random_seed_used"] = seed
        all_questions.append(q)

    for i, q in enumerate(gsm_questions):
        q["question_identifier"] = f"gsm8k_{i:04d}"
        q["source_dataset"] = "gsm8k"
        q["subject_category"] = "mathematics_word_problem"
        q["question_text"] = q["question"]
        q["answer_options"] = None
        q["random_seed_used"] = seed
        all_questions.append(q)

    # Mitigation subset
    mitigation_seed = seed + 1
    if dry_run:
        for q in all_questions:
            q["included_in_mitigation_subset"] = True
    else:
        all_questions = select_mitigation_subset(all_questions, mitigation_seed)

    # Build DataFrame with exact column names from spec
    rows = []
    for q in all_questions:
        rows.append({
            "question_identifier": q["question_identifier"],
            "source_dataset": q["source_dataset"],
            "subject_category": q["subject_category"],
            "question_text": q["question_text"],
            "answer_options": json.dumps(q["answer_options"]) if q["answer_options"] else None,
            "correct_answer": q["correct_answer"],
            "correct_answer_full_text": q.get("correct_answer_full_text", q["correct_answer"]),
            "wrong_answer_pool": json.dumps(q["wrong_answer_pool"]),
            "difficulty_stratum": q.get("difficulty_stratum", "unknown"),
            "random_seed_used": q["random_seed_used"],
            "included_in_mitigation_subset": q.get("included_in_mitigation_subset", False),
        })

    df = pd.DataFrame(rows)

    # Save
    output_path = paths["question_pool_file"]
    if not Path(output_path).is_absolute():
        output_path = project_root / output_path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(output_path), index=False)

    # ── Post-build sanity assertions ──────────────────────────────────────────
    if not dry_run:
        expected_total = exp_config["total_questions_in_pool"]
        expected_mmlu  = exp_config["mmlu_pro_question_count"]
        expected_gsm   = exp_config["gsm8k_question_count"]
        actual_mmlu    = int((df["source_dataset"] == "mmlu_pro").sum())
        actual_gsm     = int((df["source_dataset"] == "gsm8k").sum())
        actual_total   = len(df)

        if actual_mmlu != expected_mmlu:
            raise ValueError(
                f"Question pool MMLU-Pro count mismatch: got {actual_mmlu}, "
                f"expected {expected_mmlu}. Check subject names and dataset availability."
            )
        if actual_gsm != expected_gsm:
            raise ValueError(
                f"Question pool GSM8K count mismatch: got {actual_gsm}, "
                f"expected {expected_gsm}."
            )
        if actual_total != expected_total:
            raise ValueError(
                f"Question pool total mismatch: got {actual_total}, "
                f"expected {expected_total}."
            )

    logger.info(f"Question pool saved: {len(df)} questions to {output_path}")
    logger.info(f"  MMLU-Pro: {len(df[df['source_dataset'] == 'mmlu_pro'])}")
    logger.info(f"  GSM8K: {len(df[df['source_dataset'] == 'gsm8k'])}")
    logger.info(f"  Mitigation subset: {df['included_in_mitigation_subset'].sum()}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project_root = Path(__file__).parent.parent.resolve()
    load_dotenv(project_root / ".env")
    build_question_pool(project_root, probe_agent=None)
