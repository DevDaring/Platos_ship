"""
environment_check.py — Pre-flight verification for the experiment pipeline.

Checks: Python version, CUDA/GPU, flash_attn, API keys, paths, disk space,
dataset column schemas. Exits non-zero on any failure.
"""

import os
import sys
import json
import shutil
import logging
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml
from dotenv import load_dotenv

logger = logging.getLogger("platos_ship.environment")


def resolve_paths(project_root: Path) -> Dict[str, Path]:
    """Load paths.yaml and resolve all paths relative to project_root."""
    paths_file = project_root / "config" / "paths.yaml"
    with open(paths_file, "r") as f:
        raw_paths = yaml.safe_load(f)

    resolved = {}
    for key, value in raw_paths.items():
        p = Path(os.path.expanduser(os.path.expandvars(value)))
        if not p.is_absolute():
            p = project_root / p
        resolved[key] = p.resolve()
    return resolved


def check_python_version() -> Tuple[bool, str]:
    """Verify Python 3.12.x is active."""
    v = sys.version_info
    if v.major == 3 and v.minor == 12:
        return True, f"Python {v.major}.{v.minor}.{v.micro}"
    return False, f"Expected Python 3.12, got {v.major}.{v.minor}.{v.micro}"


def check_cuda_and_gpu() -> Tuple[bool, str]:
    """Verify CUDA availability and A100 GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() returned False"
        device_name = torch.cuda.get_device_name(0)
        total_mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        free_mem = torch.cuda.mem_get_info(0)
        free_gb = free_mem[0] / (1024 ** 3)
        msg = f"{device_name}, {total_mem_gb:.1f} GB total, {free_gb:.1f} GB free"
        if free_gb < 35:
            return False, f"Insufficient free VRAM: {msg}"
        return True, msg
    except Exception as e:
        return False, f"CUDA check failed: {e}"


def check_flash_attention() -> Tuple[bool, str]:
    """Verify flash_attn imports."""
    try:
        import flash_attn
        return True, f"flash_attn version {flash_attn.__version__}"
    except ImportError as e:
        return False, f"flash_attn import failed: {e}"


def check_api_keys_deepseek() -> Tuple[bool, str]:
    """Validate DeepSeek API keys by calling model listing endpoint."""
    from openai import OpenAI

    base_url = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1")
    results = []
    all_models = set()

    for var in ["DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2"]:
        key = os.getenv(var, "").strip()
        if not key:
            results.append(f"{var}: MISSING")
            continue
        try:
            client = OpenAI(api_key=key, base_url=base_url, timeout=30)
            models = client.models.list()
            model_names = [m.id for m in models.data]
            all_models.update(model_names)
            results.append(f"{var}: OK ({len(model_names)} models)")
        except Exception as e:
            results.append(f"{var}: FAILED ({e})")

    # Log available models
    primary = os.getenv("DEEPSEEK_PRIMARY_MODEL_NAME", "deepseek-chat")
    judge = os.getenv("DEEPSEEK_JUDGE_MODEL_NAME", "deepseek-chat")
    model_check = f"Primary '{primary}' {'FOUND' if primary in all_models else 'NOT FOUND'}, "
    model_check += f"Judge '{judge}' {'FOUND' if judge in all_models else 'NOT FOUND'}"

    ok = all("OK" in r for r in results)
    return ok, "; ".join(results) + f" | {model_check} | All models: {sorted(all_models)}"


def check_api_keys_openrouter() -> Tuple[bool, str]:
    """Validate OpenRouter API keys and verify all required model names are available."""
    import requests

    results = []
    all_model_ids = set()

    for var in ["OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2"]:
        key = os.getenv(var, "").strip()
        if not key:
            results.append(f"{var}: MISSING")
            continue
        try:
            resp = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                model_ids = [m["id"] for m in data.get("data", [])]
                all_model_ids.update(model_ids)
                results.append(f"{var}: OK ({len(model_ids)} models available)")
            else:
                results.append(f"{var}: HTTP {resp.status_code}")
        except Exception as e:
            results.append(f"{var}: FAILED ({e})")

    # Verify all three required model names are present
    models_to_check = {
        "smart": os.getenv("OPENROUTER_PRIMARY_MODEL_NAME", "openai/gpt-4o-mini"),
        "llama_dumb": os.getenv("OPENROUTER_LLAMA_MODEL_NAME", "meta-llama/llama-3.1-8b-instruct"),
        "gemma_dumb": os.getenv("OPENROUTER_GEMMA_MODEL_NAME", "google/gemma-3-4b-it"),
    }
    for role, model_id in models_to_check.items():
        found = model_id in all_model_ids
        results.append(f"{role} model '{model_id}': {'FOUND' if found else 'NOT FOUND'}")

    ok = all("MISSING" not in r and "FAILED" not in r and "HTTP" not in r
             and "NOT FOUND" not in r for r in results)
    return ok, " | ".join(results)


def check_api_keys_gemini() -> Tuple[bool, str]:
    """Validate Gemini API keys."""
    results = []
    for i in range(1, 5):
        var = f"GEMINI_API_KEY_{i}"
        key = os.getenv(var, "").strip()
        if not key:
            results.append(f"{var}: MISSING")
            continue
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            model = genai.GenerativeModel(os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash"))
            resp = model.generate_content("Say OK", generation_config={"max_output_tokens": 5})
            results.append(f"{var}: OK")
        except Exception as e:
            results.append(f"{var}: FAILED ({e})")

    ok = all("OK" in r for r in results)
    return ok, "; ".join(results)


def check_api_keys_mistral() -> Tuple[bool, str]:
    """Validate Mistral API key."""
    key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not key:
        return False, "MISTRAL_API_KEY: MISSING"
    try:
        from mistralai import Mistral
        client = Mistral(api_key=key)
        model_name = os.getenv("MISTRAL_MODEL_NAME", "mistral-small-latest")
        resp = client.chat.complete(
            model=model_name,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        return True, f"MISTRAL_API_KEY: OK, model={model_name}"
    except Exception as e:
        return False, f"MISTRAL_API_KEY: FAILED ({e})"


def check_huggingface_token() -> Tuple[bool, str]:
    """Validate HuggingFace token and model access."""
    token = os.getenv("HUGGINGFACE_TOKEN", "").strip()
    if not token:
        return False, "HUGGINGFACE_TOKEN: MISSING"
    try:
        from huggingface_hub import whoami, model_info

        user_info = whoami(token=token)
        username = user_info.get("name", "unknown")

        repos_to_check = [
            "meta-llama/Llama-3.1-8B-Instruct",
            "google/gemma-3-4b-it",
        ]
        access_results = []
        for repo in repos_to_check:
            try:
                info = model_info(repo, token=token)
                access_results.append(f"{repo}: accessible")
            except Exception:
                access_results.append(f"{repo}: NO ACCESS")

        return True, f"User={username}, {', '.join(access_results)}"
    except Exception as e:
        return False, f"HuggingFace check failed: {e}"


def check_paths(project_root: Path) -> Tuple[bool, str]:
    """Verify all paths exist or can be created."""
    resolved = resolve_paths(project_root)
    results = []
    all_ok = True

    for key, path in resolved.items():
        if key.endswith("_file"):
            # For file paths, check parent directory
            parent = path.parent
            try:
                parent.mkdir(parents=True, exist_ok=True)
                results.append(f"{key}: parent dir OK ({parent})")
            except Exception as e:
                results.append(f"{key}: FAILED to create parent ({e})")
                all_ok = False
        elif key.endswith("_directory") or key.endswith("_root"):
            try:
                path.mkdir(parents=True, exist_ok=True)
                results.append(f"{key}: OK ({path})")
            except Exception as e:
                results.append(f"{key}: FAILED ({e})")
                all_ok = False
        else:
            # General path
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                results.append(f"{key}: OK ({path})")
            except Exception as e:
                results.append(f"{key}: FAILED ({e})")
                all_ok = False

    return all_ok, "\n  ".join(results)


def check_disk_space(project_root: Path) -> Tuple[bool, str]:
    """Verify at least 30 GB free on data partition."""
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(str(data_dir))
    free_gb = usage.free / (1024 ** 3)
    if free_gb >= 30:
        return True, f"{free_gb:.1f} GB free"
    return False, f"Only {free_gb:.1f} GB free (need 30 GB)"


def check_dataset_schemas() -> Tuple[bool, str]:
    """Load small samples of MMLU-Pro and GSM8K to verify column schemas."""
    try:
        from datasets import load_dataset

        # MMLU-Pro
        mmlu = load_dataset("TIGER-Lab/MMLU-Pro", split="test", streaming=True)
        sample = next(iter(mmlu))
        mmlu_cols = set(sample.keys())
        expected_mmlu = {"question", "options", "answer", "category"}
        missing_mmlu = expected_mmlu - mmlu_cols
        mmlu_msg = f"MMLU-Pro columns: {sorted(mmlu_cols)}"
        if missing_mmlu:
            mmlu_msg += f" | MISSING: {missing_mmlu}"

        # GSM8K
        gsm = load_dataset("gsm8k", "main", split="test", streaming=True)
        sample = next(iter(gsm))
        gsm_cols = set(sample.keys())
        expected_gsm = {"question", "answer"}
        missing_gsm = expected_gsm - gsm_cols
        gsm_msg = f"GSM8K columns: {sorted(gsm_cols)}"
        if missing_gsm:
            gsm_msg += f" | MISSING: {missing_gsm}"

        ok = not missing_mmlu and not missing_gsm
        return ok, f"{mmlu_msg} | {gsm_msg}"
    except Exception as e:
        return False, f"Dataset schema check failed: {e}"


def run_all_checks(project_root: Path) -> Dict[str, Any]:
    """Run all pre-flight checks and return results."""
    load_dotenv(project_root / ".env")

    checks = {
        "python_version": check_python_version,
        "deepseek_api_keys": check_api_keys_deepseek,
        "openrouter_api_keys": check_api_keys_openrouter,
        "gemini_api_keys": check_api_keys_gemini,
        "mistral_api_key": check_api_keys_mistral,
        "paths": lambda: check_paths(project_root),
        "disk_space": lambda: check_disk_space(project_root),
        "dataset_schemas": check_dataset_schemas,
    }

    results = {}
    all_passed = True

    for name, check_fn in checks.items():
        try:
            passed, message = check_fn()
        except Exception as e:
            passed, message = False, f"Exception: {e}"

        results[name] = {"passed": passed, "message": message}
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"[{status}] {name}: {message}")

        if not passed:
            all_passed = False

    results["all_passed"] = all_passed
    return results


def main():
    """Entry point for environment check script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    project_root = Path(__file__).parent.parent.resolve()
    logger.info(f"Running environment checks from: {project_root}")

    results = run_all_checks(project_root)

    # Write results to logs
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    with open(logs_dir / "environment_check_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Write available DeepSeek models to separate log
    ds_msg = results.get("deepseek_api_keys", {}).get("message", "")
    if "All models:" in ds_msg:
        models_part = ds_msg.split("All models: ")[1]
        with open(logs_dir / "available_deepseek_models.log", "w") as f:
            f.write(f"Available DeepSeek models:\n{models_part}\n")

    if results["all_passed"]:
        logger.info("=" * 60)
        logger.info("ALL PRE-FLIGHT CHECKS PASSED")
        logger.info("=" * 60)
        return 0
    else:
        failed = [k for k, v in results.items() if isinstance(v, dict) and not v.get("passed", True)]
        logger.error("=" * 60)
        logger.error(f"PRE-FLIGHT CHECKS FAILED: {failed}")
        logger.error("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
