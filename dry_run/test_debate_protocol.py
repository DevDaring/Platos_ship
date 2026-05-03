"""
test_debate_protocol.py — Tests debate protocol for all 5 conditions.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv


class MockSmartAgent:
    """Mock smart agent that returns structured responses."""
    provider = "mock"
    model_name = "mock-smart"
    actually_loaded_repository = "mock-smart-model"

    def __init__(self, name="mock_smart"):
        self.agent_name = name
        self._call_count = 0

    def generate_response(self, system_prompt, user_prompt, temperature, maximum_output_tokens, request_metadata=None):
        from src.agent_wrappers.base_agent import AgentResponse
        self._call_count += 1
        # Alternate between correct and wrong answers for testing
        if self._call_count % 3 == 0:
            text = "After analysis, I believe the answer is B.\nFinal answer: B\nConfidence: 75"
        else:
            text = "The correct answer is A based on reasoning.\nFinal answer: A\nConfidence: 85"
        return AgentResponse(
            raw_text_output=text,
            wall_clock_latency_seconds=0.1,
            total_input_tokens=50,
            total_output_tokens=30,
            model_name_returned_by_provider="mock-smart",
            error_status="success",
        )

    @property
    def usage_summary(self):
        return {"total_calls": self._call_count}


class MockDumbAgent:
    """Mock dumb agent."""
    provider = "mock_local"
    actually_loaded_repository = "mock-dumb-model"

    def __init__(self, name="mock_dumb"):
        self.agent_name = name
        self._call_count = 0

    def generate_response(self, system_prompt, user_prompt, temperature, maximum_output_tokens, request_metadata=None):
        from src.agent_wrappers.base_agent import AgentResponse
        self._call_count += 1
        # Dumb agent always picks wrong answer with high confidence
        text = "Clearly the answer is C because of keyword matching.\nFinal answer: C\nConfidence: 90"
        return AgentResponse(
            raw_text_output=text,
            wall_clock_latency_seconds=0.05,
            total_input_tokens=30,
            total_output_tokens=20,
            model_name_returned_by_provider="mock-dumb",
            error_status="success",
        )

    @property
    def usage_summary(self):
        return {"total_calls": self._call_count}


class MockLowConfDumbAgent:
    """Mock dumb agent with low confidence (for C5 empty-peers test)."""
    provider = "mock_local"
    actually_loaded_repository = "mock-dumb-lowconf"

    def __init__(self, name="mock_dumb_low"):
        self.agent_name = name
        self._call_count = 0

    def generate_response(self, system_prompt, user_prompt, temperature, maximum_output_tokens, request_metadata=None):
        from src.agent_wrappers.base_agent import AgentResponse
        self._call_count += 1
        text = "Maybe C? Not sure.\nFinal answer: C\nConfidence: 30"
        return AgentResponse(
            raw_text_output=text,
            wall_clock_latency_seconds=0.05,
            total_input_tokens=30,
            total_output_tokens=20,
            model_name_returned_by_provider="mock-dumb-lowconf",
            error_status="success",
        )


def test_c1_solo():
    """Test C1: Single smart agent, no debate."""
    from src.debate_protocol import run_round0
    from src.agent_wrappers.judge_agent import JudgeCascade

    print("  Testing C1 (smart solo)...")
    judge = JudgeCascade()
    smart = MockSmartAgent()

    question = {
        "question_identifier": "test_c1_001",
        "question_text": "What is the capital of France?",
        "answer_options": '["London", "Paris", "Berlin", "Madrid"]',
        "correct_answer": "B",
    }

    agents = [{"identifier": "Agent_Alpha", "role": "smart_focal", "agent": smart,
               "provider": "mock", "temperature": 0.7, "max_tokens": 600}]

    r0 = run_round0(agents, question, judge)
    assert len(r0) == 1, f"Expected 1 response, got {len(r0)}"
    assert r0[0]["extracted_final_answer"] is not None
    print(f"  R0 answer: {r0[0]['extracted_final_answer']}, confidence: {r0[0].get('extracted_self_reported_confidence_integer')}")
    print("  ✓ C1 passed")
    return True


def test_c4_debate():
    """Test C4: 1 smart + 2 dumb, standard debate."""
    from src.debate_protocol import run_round0, run_round1_standard
    from src.agent_wrappers.judge_agent import JudgeCascade

    print("  Testing C4 (1 smart + 2 dumb)...")
    judge = JudgeCascade()
    smart = MockSmartAgent()
    dumb1 = MockDumbAgent("dumb_llama")
    dumb2 = MockDumbAgent("dumb_gemma")

    question = {
        "question_identifier": "test_c4_001",
        "question_text": "What is 2+2?",
        "answer_options": '["3", "4", "5", "6"]',
        "correct_answer": "B",
    }

    agents = [
        {"identifier": "Agent_Alpha", "role": "smart_focal", "agent": smart,
         "provider": "mock", "temperature": 0.7, "max_tokens": 600},
        {"identifier": "Agent_Dumb_1", "role": "dumb", "agent": dumb1,
         "provider": "mock_local", "temperature": 0.9, "max_tokens": 350,
         "persona_text": "The answer is 5 because 2+2=5.\nFinal answer: C\nConfidence: 95"},
        {"identifier": "Agent_Dumb_2", "role": "dumb", "agent": dumb2,
         "provider": "mock_local", "temperature": 0.9, "max_tokens": 350,
         "persona_text": "Clearly 6 is correct.\nFinal answer: D\nConfidence: 88"},
    ]

    r0 = run_round0(agents, question, judge)
    assert len(r0) == 3, f"Expected 3 R0 responses, got {len(r0)}"

    r1 = run_round1_standard(agents, r0, question, judge, ordering_seed=42)
    assert len(r1) == 3, f"Expected 3 R1 responses, got {len(r1)}"

    print(f"  R0 focal: {r0[0]['extracted_final_answer']}")
    print(f"  R1 focal: {r1[0]['extracted_final_answer']}")
    print("  ✓ C4 passed")
    return True


def test_c5_confidence_weighted():
    """Test C5: confidence-weighted with filtered peers."""
    from src.debate_protocol import run_round0
    from src.confidence_weighted_protocol import run_round1_confidence_weighted
    from src.agent_wrappers.judge_agent import JudgeCascade

    print("  Testing C5 (confidence-weighted)...")
    judge = JudgeCascade()
    smart = MockSmartAgent()
    dumb1 = MockDumbAgent("dumb1")
    dumb2 = MockDumbAgent("dumb2")

    question = {
        "question_identifier": "test_c5_001",
        "question_text": "What is the derivative of x^2?",
        "answer_options": '["x", "2x", "x^2", "2"]',
        "correct_answer": "B",
    }

    agents = [
        {"identifier": "Agent_Alpha", "role": "smart_focal", "agent": smart,
         "provider": "mock", "temperature": 0.7, "max_tokens": 600},
        {"identifier": "Agent_Dumb_1", "role": "dumb", "agent": dumb1,
         "provider": "mock_local", "temperature": 0.9, "max_tokens": 350,
         "persona_text": "The answer is x.\nFinal answer: A\nConfidence: 90"},
        {"identifier": "Agent_Dumb_2", "role": "dumb", "agent": dumb2,
         "provider": "mock_local", "temperature": 0.9, "max_tokens": 350,
         "persona_text": "It's x^2.\nFinal answer: C\nConfidence: 70"},
    ]

    r0 = run_round0(agents, question, judge)

    focal_info = agents[0]
    dumb_infos = agents[1:]

    r1, filtered_count = run_round1_confidence_weighted(
        focal_info, dumb_infos, r0, question, judge, ordering_seed=42, confidence_threshold=60
    )

    print(f"  Filtered out: {filtered_count}")
    print(f"  R1 focal: {r1[0]['extracted_final_answer']}")
    print("  ✓ C5 passed")
    return True


def test_c5_empty_peers():
    """Test C5 edge case: both peers below confidence threshold."""
    from src.debate_protocol import run_round0
    from src.confidence_weighted_protocol import run_round1_confidence_weighted
    from src.agent_wrappers.judge_agent import JudgeCascade

    print("  Testing C5 empty-peers edge case...")
    judge = JudgeCascade()
    smart = MockSmartAgent()
    dumb1 = MockLowConfDumbAgent("dumb_low1")
    dumb2 = MockLowConfDumbAgent("dumb_low2")

    question = {
        "question_identifier": "test_c5_empty_001",
        "question_text": "What color is the sky?",
        "answer_options": '["Red", "Blue", "Green", "Yellow"]',
        "correct_answer": "B",
    }

    agents = [
        {"identifier": "Agent_Alpha", "role": "smart_focal", "agent": smart,
         "provider": "mock", "temperature": 0.7, "max_tokens": 600},
        {"identifier": "Agent_Dumb_1", "role": "dumb", "agent": dumb1,
         "provider": "mock_local", "temperature": 0.9, "max_tokens": 350,
         "persona_text": "Maybe red?\nFinal answer: A\nConfidence: 20"},
        {"identifier": "Agent_Dumb_2", "role": "dumb", "agent": dumb2,
         "provider": "mock_local", "temperature": 0.9, "max_tokens": 350,
         "persona_text": "Green maybe?\nFinal answer: C\nConfidence: 15"},
    ]

    r0 = run_round0(agents, question, judge)
    focal_info = agents[0]
    dumb_infos = agents[1:]

    r1, filtered_count = run_round1_confidence_weighted(
        focal_info, dumb_infos, r0, question, judge, ordering_seed=42, confidence_threshold=60
    )

    assert filtered_count == 2, f"Expected 2 filtered out, got {filtered_count}"
    print(f"  Both peers filtered (count={filtered_count})")
    print(f"  Focal still answered: {r1[0]['extracted_final_answer']}")
    print("  ✓ Empty-peers edge case passed")
    return True


def run_all():
    project_root = Path(__file__).parent.parent.resolve()
    load_dotenv(project_root / ".env")
    sys.path.insert(0, str(project_root))

    print("=" * 60)
    print("DEBATE PROTOCOL TESTS")
    print("=" * 60)

    all_passed = True
    tests = [
        ("C1 Solo", test_c1_solo),
        ("C4 Standard Debate", test_c4_debate),
        ("C5 Confidence-Weighted", test_c5_confidence_weighted),
        ("C5 Empty Peers", test_c5_empty_peers),
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
