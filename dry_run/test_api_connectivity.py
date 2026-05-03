"""
test_api_connectivity.py — Tests every API key individually.

Validates: 2 DeepSeek keys, 2 OpenRouter keys, 4 Gemini keys,
1 Mistral key, 1 HuggingFace token.
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


def test_deepseek_keys():
    """Test each DeepSeek API key with a simple completion."""
    from openai import OpenAI

    base_url = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("DEEPSEEK_PRIMARY_MODEL_NAME", "deepseek-chat")
    results = {}

    for i, var in enumerate(["DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2"], 1):
        key = os.getenv(var, "").strip()
        if not key:
            results[var] = "MISSING"
            continue
        try:
            client = OpenAI(api_key=key, base_url=base_url, timeout=30)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say 'test OK' and nothing else."}],
                max_tokens=10, temperature=0,
            )
            text = resp.choices[0].message.content.strip()
            results[var] = f"OK — response: '{text[:50]}'"
        except Exception as e:
            results[var] = f"FAILED — {type(e).__name__}: {str(e)[:100]}"

    return results


def test_openrouter_keys():
    """Test each OpenRouter API key."""
    from openai import OpenAI

    base_url = os.getenv("OPENROUTER_API_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_PRIMARY_MODEL_NAME", "openai/gpt-4o-mini")
    results = {}

    for var in ["OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2"]:
        key = os.getenv(var, "").strip()
        if not key:
            results[var] = "MISSING"
            continue
        try:
            client = OpenAI(
                api_key=key, base_url=base_url, timeout=30,
                default_headers={
                    "HTTP-Referer": "https://platos-ship-debate-study.research",
                    "X-Title": "Platos Ship Test",
                },
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say 'test OK' and nothing else."}],
                max_tokens=10, temperature=0,
            )
            text = resp.choices[0].message.content.strip()
            results[var] = f"OK — response: '{text[:50]}'"
        except Exception as e:
            results[var] = f"FAILED — {type(e).__name__}: {str(e)[:100]}"

    return results


def test_gemini_keys():
    """Test each Gemini API key."""
    results = {}

    for i in range(1, 5):
        var = f"GEMINI_API_KEY_{i}"
        key = os.getenv(var, "").strip()
        if not key:
            results[var] = "MISSING"
            continue
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            model = genai.GenerativeModel(os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash"))
            resp = model.generate_content(
                "Say 'test OK' and nothing else.",
                generation_config={"max_output_tokens": 10, "temperature": 0},
            )
            text = resp.text.strip() if resp.text else "empty"
            results[var] = f"OK — response: '{text[:50]}'"
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            results[var] = f"FAILED — {type(e).__name__}: {str(e)[:100]}"

    return results


def test_mistral_key():
    """Test Mistral API key."""
    key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not key:
        return {"MISTRAL_API_KEY": "MISSING"}
    try:
        from mistralai import Mistral
        client = Mistral(api_key=key)
        model = os.getenv("MISTRAL_MODEL_NAME", "mistral-small-latest")
        resp = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": "Say 'test OK' and nothing else."}],
            max_tokens=10, temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        return {"MISTRAL_API_KEY": f"OK — response: '{text[:50]}'"}
    except Exception as e:
        return {"MISTRAL_API_KEY": f"FAILED — {type(e).__name__}: {str(e)[:100]}"}


def test_huggingface_token():
    """Test HuggingFace token."""
    token = os.getenv("HUGGINGFACE_TOKEN", "").strip()
    if not token:
        return {"HUGGINGFACE_TOKEN": "MISSING"}
    try:
        from huggingface_hub import whoami, model_info
        user = whoami(token=token)
        username = user.get("name", "unknown")

        model_access = {}
        for repo in ["meta-llama/Llama-3.1-8B-Instruct", "google/gemma-3-4b-it"]:
            try:
                info = model_info(repo, token=token)
                model_access[repo] = "accessible"
            except Exception as e:
                model_access[repo] = f"no access ({e})"

        return {
            "HUGGINGFACE_TOKEN": f"OK — user={username}",
            "model_access": model_access,
        }
    except Exception as e:
        return {"HUGGINGFACE_TOKEN": f"FAILED — {type(e).__name__}: {str(e)[:100]}"}


def run_all():
    project_root = Path(__file__).parent.parent.resolve()
    load_dotenv(project_root / ".env")

    print("=" * 60)
    print("API CONNECTIVITY TESTS")
    print("=" * 60)

    all_passed = True
    tests = [
        ("DeepSeek", test_deepseek_keys),
        ("OpenRouter", test_openrouter_keys),
        ("Gemini", test_gemini_keys),
        ("Mistral", test_mistral_key),
        ("HuggingFace", test_huggingface_token),
    ]

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            results = test_fn()
            for k, v in results.items():
                status = "✓" if "OK" in str(v) else "✗"
                print(f"  {status} {k}: {v}")
                if "FAILED" in str(v) or "MISSING" in str(v):
                    all_passed = False
        except Exception as e:
            print(f"  ✗ {name}: Exception — {e}")
            all_passed = False

    print("\n" + "=" * 60)
    print(f"RESULT: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 60)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_all())
