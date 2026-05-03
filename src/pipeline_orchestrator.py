"""
pipeline_orchestrator.py — Main entry point for the experiment pipeline.

Coordinates all three stages:
  Stage 1: Main experiment (C1–C4 + cross-model validation)
  Stage 2: Calibration gate
  Stage 3: Mitigation experiment (C5, if gate passes)

Also handles dry-run mode, experiment metadata, and result summaries.
"""

import os
import sys
import json
import uuid
import time
import logging
import argparse
import datetime
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import pandas as pd
from dotenv import load_dotenv

# Configure logging before any imports that use it
def setup_logging(project_root: Path):
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Main pipeline log
    file_handler = logging.FileHandler(logs_dir / "pipeline.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))

    # API failures log
    api_handler = logging.FileHandler(logs_dir / "api_failures.log")
    api_handler.setLevel(logging.WARNING)
    api_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Separate API failure logger
    api_logger = logging.getLogger("platos_ship.api_failures")
    api_logger.addHandler(api_handler)


logger = logging.getLogger("platos_ship.orchestrator")


def get_experiment_metadata(project_root: Path) -> Dict[str, Any]:
    """Gather initial experiment metadata."""
    try:
        import torch
        _torch_available = True
    except ImportError:
        torch = None
        _torch_available = False

    # Git commit hash
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(project_root),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "not_a_git_repo"

    # Package versions
    versions = {}
    for pkg in ["torch", "transformers", "flash_attn", "accelerate",
                 "datasets", "pandas", "numpy", "openai", "scipy",
                 "statsmodels", "google.generativeai", "mistralai"]:
        try:
            import importlib as _il
            mod = _il.import_module(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"

    # GPU info
    gpu_name = "no_gpu"
    gpu_mem = 0.0
    nvidia_driver = "unknown"
    cuda_version = "unknown"
    if _torch_available and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        try:
            smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            nvidia_driver = smi
        except Exception:
            pass
        cuda_version = torch.version.cuda or "unknown"

    return {
        "experiment_run_universal_unique_identifier": str(uuid.uuid4()),
        "git_commit_hash_at_run_start": git_hash,
        "python_interpreter_path": sys.executable,
        "package_versions": versions,
        "operating_system_release": platform.platform(),
        "gpu_device_name": gpu_name,
        "gpu_total_memory_gigabytes": round(gpu_mem, 2),
        "nvidia_driver_version": nvidia_driver,
        "cuda_runtime_version": cuda_version,
        "deepseek_primary_model_name_at_runtime": os.getenv("DEEPSEEK_PRIMARY_MODEL_NAME", ""),
        "deepseek_judge_model_name_at_runtime": os.getenv("DEEPSEEK_JUDGE_MODEL_NAME", ""),
        "openrouter_primary_model_name_at_runtime": os.getenv("OPENROUTER_PRIMARY_MODEL_NAME", ""),
        "gemini_judge_model_name_at_runtime": os.getenv("GEMINI_MODEL_NAME", ""),
        "mistral_judge_model_name_at_runtime": os.getenv("MISTRAL_MODEL_NAME", ""),
        "random_seed_value": int(os.getenv("RANDOM_SEED", "20260502")),
    }


def initialize_agents(project_root: Path, dry_run: bool = False):
    """Initialize all agent instances."""
    from .agent_wrappers.deepseek_agent import DeepSeekAgent
    from .agent_wrappers.openrouter_agent import OpenRouterAgent
    from .agent_wrappers.judge_agent import JudgeCascade

    with open(project_root / "config" / "models.yaml") as f:
        models_config = yaml.safe_load(f)

    # Smart agents
    smart_agents = {}

    ds_config = models_config["smart_agents"]["deepseek_primary"]
    smart_agents["deepseek_primary"] = DeepSeekAgent(
        agent_name="deepseek_primary",
        model_name=os.getenv(ds_config["model_name_env_variable"]),
        max_retries=ds_config["maximum_retry_attempts"],
        retry_backoff_seconds=ds_config["retry_backoff_seconds"],
        timeout_seconds=ds_config["request_timeout_seconds"],
    )

    or_config = models_config["smart_agents"]["openrouter_gpt4o_mini"]
    smart_agents["openrouter_gpt4o_mini"] = OpenRouterAgent(
        agent_name="openrouter_gpt4o_mini",
        model_name=os.getenv(or_config["model_name_env_variable"]),
        max_retries=or_config["maximum_retry_attempts"],
        retry_backoff_seconds=or_config["retry_backoff_seconds"],
        timeout_seconds=or_config["request_timeout_seconds"],
    )

    # Dumb agents — served via OpenRouter (round-robin across 2 keys)
    dumb_agents = {}

    llama_config = models_config["dumb_agents"]["llama_3_1_8b_instruct"]
    dumb_agents["llama_3_1_8b_instruct"] = OpenRouterAgent(
        agent_name="llama_3_1_8b_instruct",
        model_name=os.getenv(llama_config["model_name_env_variable"],
                             "meta-llama/llama-3.1-8b-instruct"),
        max_retries=llama_config["maximum_retry_attempts"],
        retry_backoff_seconds=llama_config["retry_backoff_seconds"],
        timeout_seconds=llama_config["request_timeout_seconds"],
    )

    gemma_config = models_config["dumb_agents"]["gemma_3_4b_instruct"]
    dumb_agents["gemma_3_4b_instruct"] = OpenRouterAgent(
        agent_name="gemma_3_4b_instruct",
        model_name=os.getenv(gemma_config["model_name_env_variable"],
                             "google/gemma-3-4b-it"),
        max_retries=gemma_config["maximum_retry_attempts"],
        retry_backoff_seconds=gemma_config["retry_backoff_seconds"],
        timeout_seconds=gemma_config["request_timeout_seconds"],
    )

    # Judge cascade
    judge = JudgeCascade()

    return smart_agents, dumb_agents, judge


def run_pipeline(project_root: Path, dry_run: bool = False, stage: str = "all"):
    """
    Run the full experiment pipeline.

    Args:
        project_root: Root directory.
        dry_run: If True, use minimal sample sizes.
        stage: 'all', 'main', 'calibration', 'mitigation', or 'dry_run'
    """
    from .dataset_builder import build_question_pool
    from .persona_generator import generate_all_personas
    from .persona_validator import validate_and_regenerate
    from .trial_runner import TrialRunner
    from .calibration_gate import run_calibration_gate
    from .metrics_calculator import compute_and_save_metrics
    from .statistical_analyzer import run_statistical_analysis

    with open(project_root / "config" / "experiment.yaml") as f:
        exp_config = yaml.safe_load(f)
    with open(project_root / "config" / "paths.yaml") as f:
        paths = yaml.safe_load(f)

    # Gather metadata
    metadata = get_experiment_metadata(project_root)
    metadata_path = paths["experiment_metadata_file"]
    if not Path(metadata_path).is_absolute():
        metadata_path = project_root / metadata_path
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"PLATO'S SHIP EXPERIMENT PIPELINE — {'DRY RUN' if dry_run else 'FULL RUN'}")
    logger.info(f"Stage: {stage}")
    logger.info("=" * 60)

    # Initialize agents
    logger.info("Initializing agents...")
    smart_agents, dumb_agents, judge_cascade = initialize_agents(project_root, dry_run)

    # Update metadata with actual model info (both dumb agents served via OpenRouter)
    metadata["llama_actually_loaded_repository"] = dumb_agents["llama_3_1_8b_instruct"].model_name
    metadata["gemma_actually_loaded_repository"] = dumb_agents["gemma_3_4b_instruct"].model_name
    metadata["gemma_fallback_was_triggered"] = False  # N/A — no local loading
    metadata["transformers_version_actually_used"] = "not_applicable_api_only"

    # Build question pool
    logger.info("Building question pool...")
    llama_agent = dumb_agents["llama_3_1_8b_instruct"]
    questions_df = build_question_pool(project_root, probe_agent=llama_agent, dry_run=dry_run)

    # Generate and validate personas
    logger.info("Generating dumb personas...")
    personas_df = generate_all_personas(project_root, llama_agent, dry_run=dry_run)

    logger.info("Validating personas...")
    personas_df = validate_and_regenerate(
        personas_df, llama_agent, questions_df,
        max_regeneration_attempts=3, project_root=project_root,
    )

    # Initialize trial runner
    runner = TrialRunner(
        project_root=project_root,
        smart_agents=smart_agents,
        dumb_agents=dumb_agents,
        judge_cascade=judge_cascade,
        personas_df=personas_df,
        questions_df=questions_df,
        dry_run=dry_run,
    )

    main_conditions = list(exp_config["main_conditions"].keys())
    focal_agent_name = exp_config["focal_smart_agent_assignment"]

    # ── Stage 1: Main experiment ──
    if stage in ("all", "main", "dry_run"):
        metadata["stage_1_main_experiment_started_timestamp_utc"] = datetime.datetime.utcnow().isoformat()
        logger.info("=" * 40)
        logger.info("STAGE 1: MAIN EXPERIMENT (C1–C4)")
        logger.info("=" * 40)

        if dry_run:
            # In dry run, all 5 conditions including C5
            all_conditions = main_conditions + ["C5_one_smart_two_dumb_confidence_weighted"]
            runner.run_conditions(
                conditions=all_conditions,
                focal_agent_name=focal_agent_name,
                trials_per_question=1,
                stage_name="dry_run",
            )
        else:
            # Main conditions C1–C4
            runner.run_conditions(
                conditions=main_conditions,
                focal_agent_name=focal_agent_name,
                stage_name="stage_1_main",
            )

            # Cross-model validation subset
            cross_model_size = exp_config["cross_model_validation_subset_size"]
            cross_focal = exp_config["cross_model_validation_focal_agent"]
            cross_questions = questions_df.sample(
                n=min(cross_model_size, len(questions_df)),
                random_state=exp_config["random_seed"],
            ).reset_index(drop=True)

            logger.info(f"Cross-model validation: {cross_model_size} questions with {cross_focal}")
            runner.run_conditions(
                conditions=main_conditions,
                focal_agent_name=cross_focal,
                question_filter=cross_questions,
                stage_name="stage_1_cross_model",
            )

        metadata["stage_1_main_experiment_completed_timestamp_utc"] = datetime.datetime.utcnow().isoformat()

    # ── Stage 2: Calibration gate ──
    if stage in ("all", "calibration", "dry_run"):
        logger.info("=" * 40)
        logger.info("STAGE 2: CALIBRATION GATE")
        logger.info("=" * 40)

        if dry_run:
            # Force gate to pass in dry run
            gate_result = {"gate_decision": "passed", "metric_value": 0.99}
            logger.info("DRY RUN: Calibration gate force-passed")
        else:
            gate_result = run_calibration_gate(project_root)

        metadata["stage_2_calibration_gate_decision"] = gate_result["gate_decision"]
        metadata["stage_2_calibration_gate_metric_value_at_decision"] = gate_result.get("metric_value", 0)

        # Log decision
        if gate_result["gate_decision"] == "passed":
            logger.info("CALIBRATION_GATE_PASSED — proceeding to C5")
        else:
            logger.info("CALIBRATION_GATE_FAILED — skipping C5; mitigation hypothesis not supported")

    # ── Stage 3: Mitigation experiment (C5) ──
    c5_ran = False
    if stage in ("all", "mitigation", "dry_run"):
        gate_decision = metadata.get("stage_2_calibration_gate_decision", "failed")

        if gate_decision == "passed" or dry_run:
            logger.info("=" * 40)
            logger.info("STAGE 3: MITIGATION EXPERIMENT (C5)")
            logger.info("=" * 40)

            metadata["stage_3_mitigation_experiment_started_timestamp_utc"] = datetime.datetime.utcnow().isoformat()

            # Filter to mitigation subset
            mitigation_questions = questions_df[questions_df["included_in_mitigation_subset"] == True]
            mitigation_trials = exp_config["mitigation_condition"]["C5_one_smart_two_dumb_confidence_weighted"]["trials_per_question"]

            runner.run_conditions(
                conditions=["C5_one_smart_two_dumb_confidence_weighted"],
                focal_agent_name=focal_agent_name,
                question_filter=mitigation_questions,
                trials_per_question=mitigation_trials,
                stage_name="stage_3_mitigation",
            )

            metadata["stage_3_mitigation_experiment_completed_timestamp_utc"] = datetime.datetime.utcnow().isoformat()
            c5_ran = True
        else:
            logger.info("Stage 3 skipped: calibration gate did not pass")

    metadata["stage_3_mitigation_experiment_was_run"] = c5_ran

    # ── Metrics and Statistical Analysis ──
    logger.info("=" * 40)
    logger.info("COMPUTING METRICS AND STATISTICAL ANALYSIS")
    logger.info("=" * 40)

    try:
        metrics = compute_and_save_metrics(project_root)
        stats = run_statistical_analysis(project_root, include_c5=c5_ran)

        # Log headline results
        logger.info("\n" + "=" * 60)
        logger.info("HEADLINE RESULTS")
        logger.info("=" * 60)

        if "metrics_summary" in metrics and not metrics["metrics_summary"].empty:
            for _, row in metrics["metrics_summary"].iterrows():
                logger.info(
                    f"  {row['condition_identifier']} ({row['focal_smart_agent_name']}): "
                    f"R0_acc={row['round_zero_accuracy_rate']:.3f}, "
                    f"R1_acc={row['round_one_accuracy_rate']:.3f}, "
                    f"flip_c2i={row['flip_rate_correct_to_incorrect']:.3f}"
                )

        if not stats.empty:
            for _, row in stats.iterrows():
                logger.info(
                    f"  {row.get('comparison_label', 'unknown')}: "
                    f"p={row.get('raw_p_value', 'N/A')}, "
                    f"p_corrected={row.get('bonferroni_corrected_p_value', 'N/A')}, "
                    f"sig={row.get('is_significant_at_corrected_alpha', 'N/A')}"
                )
    except Exception as e:
        logger.error(f"Metrics/stats computation failed: {e}", exc_info=True)

    # ── Finalize metadata ──
    metadata["total_trials_completed_main_conditions"] = runner._total_trials
    metadata["total_api_calls_made_to_deepseek"] = smart_agents["deepseek_primary"].usage_summary["total_calls"]
    metadata["total_api_calls_made_to_openrouter"] = (
        smart_agents["openrouter_gpt4o_mini"].usage_summary["total_calls"]
        + dumb_agents["llama_3_1_8b_instruct"].usage_summary["total_calls"]
        + dumb_agents["gemma_3_4b_instruct"].usage_summary["total_calls"]
    )
    metadata["total_local_generations_for_llama"] = dumb_agents["llama_3_1_8b_instruct"].usage_summary["total_calls"]
    metadata["total_local_generations_for_gemma"] = dumb_agents["gemma_3_4b_instruct"].usage_summary["total_calls"]
    metadata["total_judge_invocations"] = judge_cascade.usage_stats["total_judge_calls"]

    # Save metadata
    with open(str(metadata_path), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    if dry_run:
        metadata["dry_run_passed_timestamp_utc"] = datetime.datetime.utcnow().isoformat()
        with open(str(metadata_path), "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("=" * 60)
        logger.info("DRY RUN COMPLETE")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Plato's Ship Experiment Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    parser.add_argument("--stage", choices=["all", "main", "calibration", "mitigation", "dry_run"],
                        default="all", help="Which stage to run")
    parser.add_argument("--project-root", type=str, default=None,
                        help="Project root directory (defaults to parent of src/)")
    args = parser.parse_args()

    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        project_root = Path(__file__).parent.parent.resolve()

    # Load env
    load_dotenv(project_root / ".env")

    # Setup logging
    setup_logging(project_root)

    stage = args.stage
    if args.dry_run:
        stage = "dry_run"

    # Set up SMS notifier (TextBelt.py lives at project root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        import TextBelt as _notifier
        _notifier.register_exit_notifier()
        _has_notifier = True
        logger.info("[Notifier] SMS exit-notifier registered")
    except Exception as _ne:
        _has_notifier = False
        logger.warning(f"[Notifier] Could not load TextBelt notifier: {_ne}")

    try:
        run_pipeline(project_root, dry_run=args.dry_run, stage=stage)
        if _has_notifier:
            mode = "DRY RUN" if args.dry_run else f"stage={stage}"
            _notifier.send_sms(f"✅ Plato's Ship pipeline completed ({mode})")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        if _has_notifier:
            err_summary = str(e)[:250]
            _notifier.send_sms(f"❌ Plato's Ship FAILED: {err_summary}")
        sys.exit(1)


if __name__ == "__main__":
    main()
