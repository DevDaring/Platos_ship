"""
test_answer_extraction.py — Unit tests for regex parser with 20+ edge cases.
"""

import sys
from pathlib import Path


def test_answer_extraction():
    """Test regex extraction with comprehensive edge cases."""
    sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
    from src.agent_wrappers.judge_agent import extract_answer_regex, extract_confidence_regex

    # === Answer extraction tests ===
    answer_cases = [
        # (input_text, expected_answer, description)
        ("Final answer: A", "A", "Simple letter"),
        ("Final answer: B\n", "B", "Letter with newline"),
        ("Final answer: C\nConfidence: 85", "C", "Letter with confidence"),
        ("Final answer: D ", "D", "Letter with trailing space"),
        ("final answer: E", "E", "Lowercase prefix"),
        ("FINAL ANSWER: F", None, "ALL CAPS — regex is case-specific for Final/final"),
        ("Final answer: 42", "42", "Numeric answer"),
        ("Final answer: -5", "-5", "Negative number"),
        ("Final answer: 3.14", "3.14", "Decimal"),
        ("Final answer: 1,234", "1234", "Number with comma"),
        ("The answer is A", "A", "Alternative format"),
        ("The answer is B.", "B", "With period"),
        ("the answer is C", "C", "Lowercase alternative"),
        ("I believe the answer is D based on analysis", "D", "Embedded answer"),
        ("**A**", "A", "Bold letter at end"),
        ("[B]", "B", "Bracketed letter at end"),
        ("(C)", "C", "Parenthesized letter at end"),
        ("  A  ", "A", "Just a letter on its own line"),
        ("Some reasoning...\nFinal answer: G\nMore text", "G", "Answer in middle"),
        ("No clear answer here", None, "No answer pattern"),
        ("", None, "Empty string"),
        ("Random gibberish without structure", None, "Unstructured text"),
        ("I think A or B\nFinal answer: A", "A", "Multiple letters but final answer clear"),
        ("Step 1: compute...\nStep 2: therefore...\nFinal answer: H\nConfidence: 92", "H", "Multi-line with steps"),
    ]

    passed = 0
    failed = 0
    for text, expected, desc in answer_cases:
        result = extract_answer_regex(text)
        ok = result == expected
        status = "✓" if ok else "✗"
        if not ok:
            print(f"  {status} [{desc}]: expected={expected}, got={result}")
            failed += 1
        else:
            passed += 1

    print(f"  Answer extraction: {passed}/{passed+failed} passed")

    # === Confidence extraction tests ===
    confidence_cases = [
        # (input_text, expected_value, expected_status, description)
        ("Confidence: 85", 85, "success", "Simple integer"),
        ("Confidence: 0", 0, "success", "Zero"),
        ("Confidence: 100", 100, "success", "Maximum"),
        ("confidence: 50", 50, "success", "Lowercase"),
        ("Confidence: 150", 100, "out_of_range_clamped", "Above 100"),
        ("No confidence line here", None, "missing_line", "Missing"),
        ("", None, "missing_line", "Empty"),
        ("Confidence: abc", None, "missing_line", "Non-numeric — regex won't match"),
        ("Some text\nConfidence: 72\nMore text", 72, "success", "Middle of text"),
    ]

    conf_passed = 0
    conf_failed = 0
    for text, expected_val, expected_status, desc in confidence_cases:
        val, status = extract_confidence_regex(text)
        ok = val == expected_val and status == expected_status
        s = "✓" if ok else "✗"
        if not ok:
            print(f"  {s} [{desc}]: expected=({expected_val}, {expected_status}), got=({val}, {status})")
            conf_failed += 1
        else:
            conf_passed += 1

    print(f"  Confidence extraction: {conf_passed}/{conf_passed+conf_failed} passed")

    total_pass = passed + conf_passed
    total_fail = failed + conf_failed
    print(f"\n  Total: {total_pass}/{total_pass+total_fail} passed")

    return total_fail == 0


def run_all():
    print("=" * 60)
    print("ANSWER EXTRACTION TESTS")
    print("=" * 60)

    try:
        result = test_answer_extraction()
        print(f"\nRESULT: {'ALL PASSED' if result else 'SOME FAILED'}")
        return 0 if result else 1
    except Exception as e:
        print(f"\n✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all())
