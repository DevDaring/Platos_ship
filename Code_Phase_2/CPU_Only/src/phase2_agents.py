"""
phase2_agents.py — Build every agent from the Phase 2 multi-provider registry.

Replaces Phase 1's hardcoded initialize_agents(). Reads config/models.yaml
(providers registry + model blocks) and returns ready OpenAICompatibleAgent
instances, routed to whichever provider each model's `provider:` field names.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml

from .agent_wrappers.openai_compatible_agent import build_agent_from_config
from .agent_wrappers.judge_agent import JudgeCascade

logger = logging.getLogger("platos_ship.phase2_agents")


def load_models_config(project_root: Path) -> Dict[str, Any]:
    with open(project_root / "config" / "models.yaml") as f:
        return yaml.safe_load(f)


def _defaults(models_config: Dict[str, Any]) -> Dict[str, Any]:
    return models_config.get("request_defaults", {})


def _build_block(
    models_config: Dict[str, Any], block_name: str,
) -> Dict[str, Any]:
    """Build every agent in a named model block (skipping disabled entries)."""
    providers = models_config["providers"]
    defaults = _defaults(models_config)
    out = {}
    for key, spec in (models_config.get(block_name) or {}).items():
        if spec.get("enabled") is False:
            continue
        try:
            out[key] = build_agent_from_config(
                agent_name=key,
                provider_key=spec["provider"],
                model_slug=spec["model_slug"],
                providers_config=providers,
                max_retries=defaults.get("maximum_retry_attempts", 5),
                retry_backoff_seconds=defaults.get("retry_backoff_seconds", [2, 4, 8, 16, 32]),
                timeout_seconds=defaults.get("request_timeout_seconds", 120),
            )
        except Exception as e:
            logger.error(f"Could not build agent '{key}' in block '{block_name}': {e}")
            raise
    return out


def initialize_phase2_agents(
    project_root: Path,
    include_sweep: bool = False,
    include_heterogeneous: bool = False,
) -> Dict[str, Any]:
    """
    Build all agents needed for the requested experiments.

    Returns a dict with keys:
        smart_agents, dumb_agents, sweep_focal_agents,
        heterogeneous_agents, judge_cascade, models_config
    """
    models_config = load_models_config(project_root)

    smart_agents = _build_block(models_config, "smart_agents")
    dumb_agents = _build_block(models_config, "dumb_agents")

    sweep_focal_agents = {}
    if include_sweep:
        sweep_focal_agents = _build_block(models_config, "sweep_focal_agents")
        sweep_focal_agents.update(_build_block(models_config, "optional_premium_focal_agents"))

    heterogeneous_agents = {}
    if include_heterogeneous:
        heterogeneous_agents = _build_block(models_config, "heterogeneous_smart_agents")

    judge_cascade = JudgeCascade(models_config)

    logger.info(
        f"Phase 2 agents ready: {len(smart_agents)} smart, {len(dumb_agents)} weak, "
        f"{len(sweep_focal_agents)} sweep, {len(heterogeneous_agents)} heterogeneous."
    )

    return {
        "smart_agents": smart_agents,
        "dumb_agents": dumb_agents,
        "sweep_focal_agents": sweep_focal_agents,
        "heterogeneous_agents": heterogeneous_agents,
        "judge_cascade": judge_cascade,
        "models_config": models_config,
    }
