"""
test_persona_pipeline.py — Tests persona generation and validation.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv


def test_persona_generation():
    """Generate 1 persona variant for 1 question and validate."""
    import pandas as pd
    from src.persona_generator import generate_personas_for_question
    from src.persona_validator import validate_single_persona

    # Create a mock question
    question = {
        "question_identifier": "test_q_001",
        "question_text": "What is the capital of France?",
        "answer_options": '["London", "Paris", "Berlin", "Madrid", "Rome", "Dublin", "Warsaw", "Lisbon", "Vienna", "Brussels"]',
        "correct_answer": "B",
        "wrong_answer_pool": '["A", "C", "D", "E", "F", "G", "H", "I", "J"]',
    }

    # Use a mock agent that returns structured text
    class MockAgent:
        actually_loaded_repository = "mock-model"
        def generate_response(self, system_prompt, user_prompt, temperature, maximum_output_tokens, request_metadata=None):
            from src.agent_wrappers.base_agent import AgentResponse
            return AgentResponse(
                raw_text_output=(
                    "London is clearly the correct answer because it is the most famous "
                    "European capital. Everyone knows London when they think of Europe.\n"
                    "Final answer: A"
                ),
                wall_clock_latency_seconds=0.1,
                total_input_tokens=50,
                total_output_tokens=30,
                model_name_returned_by_provider="mock",
                error_status="success",
            )

    agent = MockAgent()
    personas = generate_personas_for_question(question, agent, num_variants=1, seed=42)

    assert len(personas) == 1, f"Expected 1 persona, got {len(personas)}"
    print(f"  Generated persona: {personas[0]['persona_identifier']}")
    print(f"  Text: {personas[0]['generated_persona_text'][:100]}...")

    # Validate
    passed, reason = validate_single_persona(personas[0])
    print(f"  Validation: passed={passed}, reason={reason}")

    # Schema check
    required_cols = [
        "persona_identifier", "question_identifier", "persona_variant_index",
        "assigned_wrong_answer_letter_or_value", "reasoning_style_label",
        "generated_persona_text", "generation_temperature",
    ]
    for col in required_cols:
        assert col in personas[0], f"Missing field: {col}"
    print("  ✓ Persona schema verified")

    return True


def run_all():
    project_root = Path(__file__).parent.parent.resolve()
    load_dotenv(project_root / ".env")
    sys.path.insert(0, str(project_root))

    print("=" * 60)
    print("PERSONA PIPELINE TESTS")
    print("=" * 60)

    try:
        result = test_persona_generation()
        print(f"\nRESULT: {'PASSED' if result else 'FAILED'}")
        return 0 if result else 1
    except Exception as e:
        print(f"\n✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all())
