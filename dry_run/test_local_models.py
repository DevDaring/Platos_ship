"""
test_local_models.py — Tests loading Llama 3.1 8B and Gemma 3/2 on GPU.

Verifies: flash_attention_2 active, VRAM < 24GB combined, basic generation.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv


def test_local_models():
    """Load both models and verify configuration."""
    import torch
    from src.agent_wrappers.local_huggingface_agent import LocalHuggingFaceAgent

    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem_before = torch.cuda.memory_allocated() / (1024**3)
        print(f"  VRAM before loading: {mem_before:.2f} GB")

    # Load Llama 3.1 8B
    print("\n  Loading Llama 3.1 8B...")
    llama = LocalHuggingFaceAgent(
        agent_name="test_llama",
        huggingface_repository="meta-llama/Llama-3.1-8B-Instruct",
        device="cuda:0",
    )
    llama.load_model()
    print(f"  Llama loaded: {llama.actually_loaded_repository}")

    if torch.cuda.is_available():
        mem_after_llama = torch.cuda.memory_allocated() / (1024**3)
        print(f"  VRAM after Llama: {mem_after_llama:.2f} GB")

    # Check flash attention
    if hasattr(llama._model, 'config'):
        attn = getattr(llama._model.config, '_attn_implementation', 'unknown')
        print(f"  Llama attention implementation: {attn}")
        assert attn == "flash_attention_2", f"Expected flash_attention_2, got {attn}"
        print("  ✓ Llama flash_attention_2 confirmed")

    # Test generation
    resp = llama.generate_response(
        system_prompt="You are a test assistant.",
        user_prompt="What is 2+2? Final answer:",
        temperature=0.1,
        maximum_output_tokens=20,
    )
    print(f"  Llama response: '{resp.raw_text_output[:100]}'")
    assert resp.error_status == "success", f"Llama generation failed: {resp.error_status}"
    print("  ✓ Llama generation OK")

    # Load Gemma
    print("\n  Loading Gemma 3/2...")
    gemma = LocalHuggingFaceAgent(
        agent_name="test_gemma",
        huggingface_repository="google/gemma-3-4b-it",
        fallback_huggingface_repository="google/gemma-2-2b-it",
        device="cuda:0",
    )
    gemma.load_model()
    print(f"  Gemma loaded: {gemma.actually_loaded_repository}")
    print(f"  Gemma fallback triggered: {gemma.fallback_was_triggered}")

    if torch.cuda.is_available():
        mem_after_both = torch.cuda.memory_allocated() / (1024**3)
        print(f"  VRAM after both models: {mem_after_both:.2f} GB")
        assert mem_after_both < 24, f"VRAM too high: {mem_after_both:.2f} GB (limit: 24 GB)"
        print(f"  ✓ Combined VRAM ({mem_after_both:.2f} GB) under 24 GB limit")

    # Check Gemma flash attention
    if hasattr(gemma._model, 'config'):
        attn = getattr(gemma._model.config, '_attn_implementation', 'unknown')
        print(f"  Gemma attention implementation: {attn}")

    # Test Gemma generation
    resp = gemma.generate_response(
        system_prompt="You are a test assistant.",
        user_prompt="What is 3+3? Final answer:",
        temperature=0.1,
        maximum_output_tokens=20,
    )
    print(f"  Gemma response: '{resp.raw_text_output[:100]}'")
    assert resp.error_status == "success", f"Gemma generation failed: {resp.error_status}"
    print("  ✓ Gemma generation OK")

    print("\n  ✓ All local model tests passed")
    return True


def run_all():
    project_root = Path(__file__).parent.parent.resolve()
    load_dotenv(project_root / ".env")
    sys.path.insert(0, str(project_root))

    print("=" * 60)
    print("LOCAL MODEL TESTS")
    print("=" * 60)

    try:
        result = test_local_models()
        print(f"\nRESULT: {'PASSED' if result else 'FAILED'}")
        return 0 if result else 1
    except Exception as e:
        print(f"\n✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all())
