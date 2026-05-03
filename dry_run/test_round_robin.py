"""
test_round_robin.py — Validates round-robin key cycling for all providers.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv


def test_deepseek_round_robin():
    """Send 4 requests, verify key alternation."""
    from src.agent_wrappers.deepseek_agent import DeepSeekAgent

    agent = DeepSeekAgent(agent_name="test_ds_rr")
    print(f"  DeepSeek: {agent._key_manager.key_count} keys")

    for i in range(4):
        resp = agent.generate_response(
            system_prompt="You are a test.",
            user_prompt=f"Say 'RR test {i}' only.",
            temperature=0, maximum_output_tokens=10,
        )
        print(f"    Request {i}: status={resp.error_status}, "
              f"key_stats={agent.key_usage_stats}")

    # Verify alternation
    stats = agent.key_usage_stats
    assert all(v == 2 for v in stats.values()), f"Keys not evenly distributed: {stats}"
    print("  ✓ DeepSeek round-robin verified")
    return True


def test_openrouter_round_robin():
    """Send 4 requests, verify key alternation."""
    from src.agent_wrappers.openrouter_agent import OpenRouterAgent

    agent = OpenRouterAgent(agent_name="test_or_rr")
    print(f"  OpenRouter: {agent._key_manager.key_count} keys")

    for i in range(4):
        resp = agent.generate_response(
            system_prompt="You are a test.",
            user_prompt=f"Say 'RR test {i}' only.",
            temperature=0, maximum_output_tokens=10,
        )
        print(f"    Request {i}: status={resp.error_status}, "
              f"key_stats={agent.key_usage_stats}")

    stats = agent.key_usage_stats
    assert all(v == 2 for v in stats.values()), f"Keys not evenly distributed: {stats}"
    print("  ✓ OpenRouter round-robin verified")
    return True


def test_gemini_round_robin():
    """Send 8 requests to Gemini judge, verify cycling through 4 keys."""
    from src.agent_wrappers.judge_agent import JudgeCascade
    import google.generativeai as genai
    import time

    keys = []
    for i in range(1, 5):
        k = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
        if k:
            keys.append(k)

    print(f"  Gemini: {len(keys)} keys")

    from src.agent_wrappers.base_agent import RoundRobinKeyManager
    rr = RoundRobinKeyManager(keys, "Gemini-Test")

    for i in range(8):
        key = rr.get_next_key()
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash"))
            resp = model.generate_content(
                f"Say '{i}' only.",
                generation_config={"max_output_tokens": 5, "temperature": 0},
            )
            print(f"    Request {i}: OK (key_idx={i % len(keys)})")
            time.sleep(0.3)
        except Exception as e:
            print(f"    Request {i}: FAILED ({e})")

    stats = rr.usage_stats
    assert all(v == 2 for v in stats.values()), f"Keys not evenly cycled: {stats}"
    print("  ✓ Gemini round-robin verified")
    return True


def test_deepseek_judge_round_robin():
    """Test DeepSeek as tertiary judge with round-robin."""
    from src.agent_wrappers.base_agent import RoundRobinKeyManager
    from openai import OpenAI

    keys = []
    for var in ["DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2"]:
        k = os.getenv(var, "").strip()
        if k:
            keys.append(k)

    rr = RoundRobinKeyManager(keys, "DeepSeek-Judge-Test")
    base_url = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("DEEPSEEK_JUDGE_MODEL_NAME", "deepseek-chat")

    print(f"  DeepSeek Judge: {len(keys)} keys")

    for i in range(4):
        key = rr.get_next_key()
        try:
            client = OpenAI(api_key=key, base_url=base_url, timeout=30)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"Say '{i}' only."}],
                max_tokens=5, temperature=0,
            )
            print(f"    Request {i}: OK")
        except Exception as e:
            print(f"    Request {i}: FAILED ({e})")

    stats = rr.usage_stats
    assert all(v == 2 for v in stats.values()), f"Not evenly cycled: {stats}"
    print("  ✓ DeepSeek Judge round-robin verified")
    return True


def run_all():
    project_root = Path(__file__).parent.parent.resolve()
    load_dotenv(project_root / ".env")
    sys.path.insert(0, str(project_root))

    print("=" * 60)
    print("ROUND-ROBIN KEY CYCLING TESTS")
    print("=" * 60)

    all_passed = True
    tests = [
        ("DeepSeek RR", test_deepseek_round_robin),
        ("OpenRouter RR", test_openrouter_round_robin),
        ("Gemini RR", test_gemini_round_robin),
        ("DeepSeek Judge RR", test_deepseek_judge_round_robin),
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
