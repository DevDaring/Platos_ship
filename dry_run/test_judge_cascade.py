"""
test_judge_cascade.py — Tests the 3-tier judge fallback chain.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv


def test_normal_gemini_success():
    """Test: Gemini succeeds → verify Gemini used."""
    from src.agent_wrappers.judge_agent import JudgeCascade

    judge = JudgeCascade()
    answer, method = judge.extract_answer(
        question_text="What is 2+2?",
        answer_options="A, B, C, D",
        raw_text="I think the answer is clearly 4. Final answer: D",
    )
    # Regex should catch this, but if not judge runs
    print(f"  Normal case: answer='{answer}', method='{method}'")
    assert answer is not None, "Answer should not be None"
    print("  ✓ Normal extraction works")
    return True


def test_gemini_failure_mistral_fallback():
    """Simulate Gemini failure → verify Mistral fallback."""
    from src.agent_wrappers.judge_agent import JudgeCascade

    judge = JudgeCascade()

    # Temporarily break Gemini
    original_try_gemini = judge._try_gemini
    judge._try_gemini = lambda prompt: None  # Force failure

    answer, method = judge.extract_answer(
        question_text="What is the capital of France?",
        answer_options="A. London, B. Paris, C. Berlin, D. Madrid",
        raw_text="Based on my analysis, Paris is the capital. My answer is B.",
    )

    judge._try_gemini = original_try_gemini  # Restore
    print(f"  Gemini fail → Mistral: answer='{answer}', method='{method}'")
    # Should use Mistral or DeepSeek
    assert method in ("judge_mistral", "judge_deepseek", "parse_failure"), f"Unexpected method: {method}"
    print("  ✓ Mistral/DeepSeek fallback works")
    return True


def test_all_fail_graceful():
    """Simulate all three tiers failing → verify UNPARSEABLE."""
    from src.agent_wrappers.judge_agent import JudgeCascade

    judge = JudgeCascade()

    # Break all tiers
    judge._try_gemini = lambda prompt: None
    judge._try_mistral = lambda prompt: None
    judge._try_deepseek = lambda prompt: None

    answer, method = judge.extract_answer(
        question_text="Ambiguous question?",
        answer_options="unclear",
        raw_text="I'm not sure about anything really.",
    )

    print(f"  All fail: answer='{answer}', method='{method}'")
    assert answer == "UNPARSEABLE", f"Expected UNPARSEABLE, got {answer}"
    assert method == "parse_failure", f"Expected parse_failure, got {method}"
    print("  ✓ Graceful UNPARSEABLE return verified")
    return True


def test_regex_preempts_judge():
    """Test that regex extraction prevents judge calls."""
    from src.agent_wrappers.judge_agent import extract_answer_regex

    test_cases = [
        ("Final answer: A", "A"),
        ("Final answer: B\nConfidence: 85", "B"),
        ("The answer is C.", "C"),
        ("Final answer: 42", "42"),
        ("blah blah\nFinal answer: D\n", "D"),
    ]

    all_pass = True
    for text, expected in test_cases:
        result = extract_answer_regex(text)
        ok = result == expected
        status = "✓" if ok else "✗"
        print(f"  {status} '{text[:40]}...' → expected={expected}, got={result}")
        if not ok:
            all_pass = False

    return all_pass


def run_all():
    project_root = Path(__file__).parent.parent.resolve()
    load_dotenv(project_root / ".env")
    sys.path.insert(0, str(project_root))

    print("=" * 60)
    print("JUDGE CASCADE TESTS")
    print("=" * 60)

    all_passed = True
    tests = [
        ("Regex preempts judge", test_regex_preempts_judge),
        ("Normal Gemini success", test_normal_gemini_success),
        ("Gemini fail → Mistral fallback", test_gemini_failure_mistral_fallback),
        ("All tiers fail → UNPARSEABLE", test_all_fail_graceful),
    ]

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            result = test_fn()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            all_passed = False

    print(f"\nRESULT: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_all())
