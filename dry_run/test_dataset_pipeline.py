"""
test_dataset_pipeline.py — Tests dataset download, schema, and sampling.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv


def test_dataset_download_and_schema():
    """Download sample data and verify column schemas."""
    from datasets import load_dataset

    print("  Loading MMLU-Pro sample...")
    mmlu = load_dataset("TIGER-Lab/MMLU-Pro", split="test", streaming=True)
    mmlu_sample = next(iter(mmlu))
    mmlu_cols = set(mmlu_sample.keys())
    print(f"  MMLU-Pro columns: {sorted(mmlu_cols)}")

    required_mmlu = {"question", "options", "answer"}
    missing = required_mmlu - mmlu_cols
    assert not missing, f"Missing MMLU-Pro columns: {missing}"
    print("  ✓ MMLU-Pro schema verified")

    print("\n  Loading GSM8K sample...")
    gsm = load_dataset("gsm8k", "main", split="test", streaming=True)
    gsm_sample = next(iter(gsm))
    gsm_cols = set(gsm_sample.keys())
    print(f"  GSM8K columns: {sorted(gsm_cols)}")

    required_gsm = {"question", "answer"}
    missing = required_gsm - gsm_cols
    assert not missing, f"Missing GSM8K columns: {missing}"
    print("  ✓ GSM8K schema verified")

    return True


def test_question_pool_build():
    """Build a minimal question pool (2 questions)."""
    from src.dataset_builder import build_question_pool
    import pandas as pd

    project_root = Path(__file__).parent.parent.resolve()
    df = build_question_pool(project_root, probe_agent=None, dry_run=True)

    print(f"\n  Question pool: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    # Verify required columns
    required_cols = [
        "question_identifier", "source_dataset", "subject_category",
        "question_text", "correct_answer", "wrong_answer_pool",
        "difficulty_stratum", "random_seed_used", "included_in_mitigation_subset",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    print("  ✓ All required columns present")

    assert len(df) >= 2, f"Expected at least 2 rows, got {len(df)}"
    print("  ✓ Question pool has sufficient rows")

    return True


def run_all():
    project_root = Path(__file__).parent.parent.resolve()
    load_dotenv(project_root / ".env")
    sys.path.insert(0, str(project_root))

    print("=" * 60)
    print("DATASET PIPELINE TESTS")
    print("=" * 60)

    all_passed = True
    tests = [
        ("Dataset schema", test_dataset_download_and_schema),
        ("Question pool build", test_question_pool_build),
    ]

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            result = test_fn()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print(f"\nRESULT: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_all())
