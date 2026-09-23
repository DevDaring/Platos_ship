"""
agents.py — build every Phase-3 agent from config/models.yaml.

A focal response is the measured unit, so it must come from the pinned
checkpoint. Phase 2 lost that guarantee: a fallback chain quietly served 75
"GPT-4o-mini" responses from gpt-4.1-mini.

The fix is not to ban fallbacks — a single-account provider like LinkAPI does
rate-limit, and a run that dies on a 429 is its own problem. The fix is to make
the chain unable to change the model. Focal and verifier agents are built with
`pin_model=True`, which wraps the chain in PinnedFallbackAgent: every link's
served model is checked against `expected_served_prefix`, and a link answering
with anything else is rejected and skipped. The route may change; the model
may not.

Peer and judge agents use the plain chain, because a peer message is an input
whose true generator is recorded per message rather than a measured unit.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from .agent_wrappers.judge_agent import JudgeCascade
from .agent_wrappers.openai_compatible_agent import FallbackAgent, build_agent_from_config
from .pinned_fallback import PinnedFallbackAgent

logger = logging.getLogger("platos_ship3.agents")


def load_models_config(project_root: Path) -> Dict[str, Any]:
    with open(Path(project_root) / "config" / "models.yaml") as handle:
        return yaml.safe_load(handle)


def _build_one(
    key: str,
    spec: Dict[str, Any],
    providers: Dict[str, Any],
    defaults: Dict[str, Any],
    allow_fallback: bool,
    pin_model: bool = False,
):
    """
    Build one agent, optionally with a fallback chain.

    `pin_model=True` wraps the chain in PinnedFallbackAgent, so a fallback can
    change the route but never the model. Focal agents always use it; that is
    what makes a cheap-but-rate-limited primary safe to keep.
    """
    backoff = defaults.get("retry_backoff_seconds", [2, 4, 8, 16, 32])
    timeout = defaults.get("request_timeout_seconds", 120)
    fallbacks = (spec.get("fallbacks") or []) if allow_fallback else []
    primary_retries = (spec.get("max_retries")
                       if spec.get("max_retries") is not None
                       else (2 if fallbacks
                             else defaults.get("maximum_retry_attempts", 5)))

    primary = build_agent_from_config(
        agent_name=key,
        provider_key=spec["provider"],
        model_slug=spec["model_slug"],
        providers_config=providers,
        max_retries=primary_retries,
        retry_backoff_seconds=backoff,
        timeout_seconds=timeout,
    )
    if not fallbacks:
        return primary

    # A fallback is optional by definition: if its provider has no key in this
    # environment, drop it and keep going rather than failing the whole run.
    chain, labels = [], []
    for index, fallback in enumerate(fallbacks):
        # Per-fallback retry policy. `max_retries: 0` means a single attempt,
        # so a slow link is skipped rather than sat in a backoff loop.
        retries = fallback.get("max_retries")
        retries = 2 if retries is None else int(retries)
        try:
            chain.append(build_agent_from_config(
                agent_name=f"{key}_fb{index + 1}",
                provider_key=fallback["provider"],
                model_slug=fallback["model_slug"],
                providers_config=providers,
                max_retries=retries,
                retry_backoff_seconds=backoff,
                timeout_seconds=timeout,
            ))
            labels.append(f"{fallback['provider']}(retries={retries})")
        except Exception as exc:
            logger.warning("'%s': fallback via %s unavailable (%s); skipping it.",
                           key, fallback["provider"], exc)

    if not chain:
        return primary

    if pin_model:
        return PinnedFallbackAgent(
            agent_name=key,
            links=[primary] + chain,
            expected_served_prefix=spec.get("expected_served_prefix"),
            labels=[f"{spec['provider']}(retries={primary_retries})"] + labels,
        )

    logger.info("'%s' fallback chain: %s -> %s",
                key, spec["provider"], " -> ".join(labels))
    return FallbackAgent(agent_name=key, primary=primary, fallbacks=chain)


class MissingCredentialsError(RuntimeError):
    """Raised when a required agent cannot be built from the environment."""


def build_agents(
    project_root: Path,
    focal_keys: list[str] | None = None,
    need_weak: bool = True,
    need_judge: bool = True,
    need_verifier: bool = False,
    skip_unavailable: bool = False,
) -> Dict[str, Any]:
    """
    Build only what the requested experiments need.

    Every focal model that cannot be built is collected and reported together,
    rather than aborting on the first one, because the usual cause is a single
    missing provider key and the useful message names all of them at once.

    `skip_unavailable=True` downgrades that to a warning and records the
    omission in `unavailable_focal`. Use it for development only: an X1 matrix
    missing a model is precisely the unequal coverage this phase exists to fix,
    so the omission is written into the run metadata where it cannot be
    forgotten.

    Returns {focal_agents, focal_specs, weak_agents, weak_specs,
             judge_cascade, verifier_agent, verifier_spec, models_config,
             unavailable_focal}.
    """
    models_config = load_models_config(project_root)
    providers = models_config["providers"]
    defaults = models_config.get("request_defaults", {})

    focal_specs = models_config.get("focal_agents") or {}
    wanted = list(focal_specs) if focal_keys is None else list(focal_keys)

    focal_agents: Dict[str, Any] = {}
    unavailable: Dict[str, str] = {}
    for key in wanted:
        if key not in focal_specs:
            raise KeyError(f"Unknown focal agent '{key}' (config/models.yaml).")
        try:
            # A focal model MAY have a fallback chain, but `pin_model=True`
            # means every link is checked against expected_served_prefix, so a
            # fallback can change the route and never the model.
            focal_agents[key] = _build_one(
                key, focal_specs[key], providers, defaults,
                allow_fallback=True, pin_model=True,
            )
        except Exception as exc:
            unavailable[key] = f"{focal_specs[key].get('provider')}: {exc}"

    if unavailable:
        lines = "\n".join(f"  - {key}: {reason}"
                          for key, reason in sorted(unavailable.items()))
        message = (
            f"{len(unavailable)} focal model(s) could not be built:\n{lines}\n"
            f"Each needs its provider's keys in .env (see .env.example). To "
            f"route a model to a provider you already hold keys for, edit its "
            f"`provider`/`model_slug` in config/models.yaml. A `fallbacks:` "
            f"chain is allowed on a focal model — it is pinned to "
            f"`expected_served_prefix`, so it can change the route but not the "
            f"model."
        )
        if not skip_unavailable:
            raise MissingCredentialsError(message)
        logger.warning("%s\nContinuing with an INCOMPLETE focal set.", message)

    weak_specs = models_config.get("weak_agents") or {}
    weak_agents = {}
    if need_weak:
        for key, spec in weak_specs.items():
            weak_agents[key] = _build_one(
                key, spec, providers, defaults, allow_fallback=True
            )

    judge_cascade = JudgeCascade(models_config) if need_judge else None

    verifier_agent, verifier_spec = None, models_config.get("verifier_agent") or {}
    if need_verifier and verifier_spec:
        verifier_agent = _build_one(
            "verifier", verifier_spec, providers, defaults,
            allow_fallback=True, pin_model=True,
        )

    logger.info(
        "Phase 3 agents ready: %d focal, %d weak, judge=%s, verifier=%s",
        len(focal_agents), len(weak_agents),
        judge_cascade is not None, verifier_agent is not None,
    )
    return {
        "focal_agents": focal_agents,
        "focal_specs": focal_specs,
        "weak_agents": weak_agents,
        "weak_specs": weak_specs,
        "judge_cascade": judge_cascade,
        "verifier_agent": verifier_agent,
        "verifier_spec": verifier_spec,
        "models_config": models_config,
        "unavailable_focal": unavailable,
    }


def resolve_focal_selector(selector: Any, focal_specs: Dict[str, Any]) -> list[str]:
    """
    Turn an experiment's `focal:` field into a concrete list of keys.

    "ALL"      -> every focal model
    "TIER_X2"  -> the models flagged `in_x2: true` (one per capability tier)
    "TIER_X3"  -> the models flagged `in_x3: true`
    [a, b]     -> used as given
    """
    if isinstance(selector, str):
        if selector == "ALL":
            return list(focal_specs)
        if selector == "TIER_X2":
            return [k for k, v in focal_specs.items() if v.get("in_x2")]
        if selector == "TIER_X3":
            return [k for k, v in focal_specs.items() if v.get("in_x3")]
        return [selector]
    return list(selector or [])
