"""
run_all_dry_tests.py — Master script that runs all dry_run tests in sequence.
"""

import sys
import time
import importlib
from pathlib import Path
from dotenv import load_dotenv


def run_all():
    project_root = Path(__file__).parent.parent.resolve()
    load_dotenv(project_root / ".env")
    sys.path.insert(0, str(project_root))

    test_modules = [
        ("1. Answer Extraction (regex)", "dry_run.test_answer_extraction"),
        ("2. Persona Pipeline (mock)", "dry_run.test_persona_pipeline"),
        ("3. API Connectivity", "dry_run.test_api_connectivity"),
        ("4. Round-Robin Key Cycling", "dry_run.test_round_robin"),
        ("5. Judge Cascade", "dry_run.test_judge_cascade"),
        ("6. Dataset Pipeline", "dry_run.test_dataset_pipeline"),
        ("7. Debate Protocol (mock)", "dry_run.test_debate_protocol"),
        ("8. Metrics & Statistics", "dry_run.test_metrics_and_stats"),
        ("9. Local Models (GPU)", "dry_run.test_local_models"),
    ]

    print("=" * 70)
    print("  PLATO'S SHIP — DRY RUN TEST SUITE")
    print("  Running all tests in sequence")
    print("=" * 70)

    results = {}
    start_time = time.time()

    for name, module_path in test_modules:
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")

        try:
            mod = importlib.import_module(module_path)
            exit_code = mod.run_all()
            results[name] = "PASSED" if exit_code == 0 else "FAILED"
        except Exception as e:
            print(f"\n  ✗ EXCEPTION: {type(e).__name__}: {e}")
            results[name] = f"ERROR: {e}"

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("  DRY RUN TEST SUITE — SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, result in results.items():
        status = "✓" if result == "PASSED" else "✗"
        print(f"  {status} {name}: {result}")
        if result != "PASSED":
            all_passed = False

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_all())
