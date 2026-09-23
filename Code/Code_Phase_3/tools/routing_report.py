#!/usr/bin/env python3
"""
routing_report.py — which account pays for which call.

Every model in the study is reached through a provider account, and the
pipeline rotates round-robin across the keys a provider lists. Before a run
that costs money it is worth seeing, in one table, which account carries which
traffic and roughly how much of it.

    python3 tools/routing_report.py              # table only, no API calls
    python3 tools/routing_report.py --probe      # one ~20-token call per
                                                 # provider to confirm the key
                                                 # works and to record the
                                                 # model string it serves

`--probe` is the check that would have caught the Phase-2 contamination on day
one: it prints the served model string next to the pinned prefix, so a
provider quietly substituting a different checkpoint is visible before 15,000
trials are run on it.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("platos_ship3.routing_report")

# Indicative unit prices (USD per 1M tokens, input/output) for the estimate
# column. Refresh from the provider pages before a large run; these move.
PRICES: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("deepseek", "deepseek-v4-flash"): (0.14, 0.28),
    ("linkapi", "gpt-4o-mini"): (0.011, 0.042),
    ("openrouter", "openai/gpt-4o-mini"): (0.15, 0.60),
    ("openrouter", "meta-llama/llama-3.1-70b-instruct"): (0.40, 0.40),
    ("openrouter", "qwen/qwen-2.5-72b-instruct"): (0.36, 0.40),
    ("openrouter", "google/gemma-3-27b-it"): (0.08, 0.16),
    ("openrouter", "mistralai/mistral-small-3.2-24b-instruct"): (0.075, 0.20),
    ("openrouter", "meta-llama/llama-3.1-8b-instruct"): (0.02, 0.03),
    ("openrouter", "google/gemma-3-4b-it"): (0.05, 0.10),
    ("openrouter", "google/gemini-2.5-flash-lite"): (0.10, 0.40),
    ("mistral", "mistral-small-latest"): (0.15, 0.60),
    ("linkapi_gemini", "gemini-2.5-flash"): (0.30, 2.50),
    ("gemini_direct", "gemini-2.5-flash"): (0.30, 2.50),
    ("openrouter", "google/gemini-2.5-flash"): (0.30, 2.50),
}

# nano-gpt does not publish a simple per-model rate card, and it is cheaper
# than OpenRouter for these models. Rather than invent a number, fall back to
# the OpenRouter price for the SAME model and label the total an upper bound.
PRICE_FALLBACK_PROVIDER = "openrouter"
NANOGPT_SLUG_ON_OPENROUTER = {
    "qwen/qwen-2.5-72b-instruct": "qwen/qwen-2.5-72b-instruct",
    "mistralai/mistral-small-3.2-24b-instruct":
        "mistralai/mistral-small-3.2-24b-instruct",
    "meta-llama/llama-3.1-8b-instruct": "meta-llama/llama-3.1-8b-instruct",
}

# Mean tokens per call observed across the Phase-2 logs.
MEAN_INPUT_TOKENS = 298
MEAN_OUTPUT_TOKENS = 157


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env",
                      PROJECT_ROOT.parent.parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)


def _present_keys(provider_spec: Dict[str, Any]) -> List[str]:
    """Which of a provider's declared key variables are actually set."""
    return [name for name in provider_spec.get("api_key_envs", [])
            if os.environ.get(name)]


def _mask(value: str) -> str:
    if not value:
        return "(unset)"
    return f"{value[:6]}…{value[-4:]}" if len(value) > 12 else "***"


def build_rows(models_config: Dict[str, Any],
               experiment: Dict[str, Any]) -> List[Dict[str, Any]]:
    providers = models_config["providers"]
    replicates = int(experiment.get("replicates_per_question", 3))
    n_main = int(experiment["pools"]["main300"]["n_questions"])

    rows: List[Dict[str, Any]] = []

    def add(role: str, key: str, spec: Dict[str, Any], calls: int,
            fallbacks_allowed: bool) -> None:
        provider = spec.get("provider")
        provider_spec = providers.get(provider, {})
        present = _present_keys(provider_spec)
        slug = spec.get("model_slug")
        price = PRICES.get((provider, slug))
        price_is_upper_bound = False
        if price is None:
            # Unknown provider rate: price the same model on OpenRouter, which
            # is the more expensive route, so the figure is an upper bound.
            equivalent = NANOGPT_SLUG_ON_OPENROUTER.get(slug, slug)
            price = PRICES.get((PRICE_FALLBACK_PROVIDER, equivalent))
            price_is_upper_bound = price is not None
        cost = None
        if price:
            # A judge or verifier declares its own output cap and is nowhere
            # near the generic per-call mean; use the declared cap when present.
            out_tokens = int(spec.get("max_output_tokens") or MEAN_OUTPUT_TOKENS)
            cost = calls * (MEAN_INPUT_TOKENS * price[0]
                            + out_tokens * price[1]) / 1_000_000
        rows.append({
            "role": role,
            "agent": key,
            "provider": provider,
            "model_slug": spec.get("model_slug"),
            "pinned_prefix": spec.get("expected_served_prefix", ""),
            "key_envs_declared": provider_spec.get("api_key_envs", []),
            "key_envs_present": present,
            "n_accounts": len(present),
            "fallbacks": ([f["provider"] for f in spec.get("fallbacks", [])]
                          if fallbacks_allowed else []),
            "est_calls": calls,
            "est_cost_usd": cost,
            "cost_is_upper_bound": price_is_upper_bound,
        })

    # Focal models: one cached Round-0 call plus the revision conditions each
    # experiment assigns. X1 (5 conditions x 8 models) dominates.
    from src.agents import resolve_focal_selector

    focal_specs = models_config.get("focal_agents", {})
    conditions_per_focal: Dict[str, int] = {}
    for spec in experiment["experiments"].values():
        if not spec.get("enabled") or spec.get("offline"):
            continue
        pool = experiment["pools"].get(spec.get("pool", "main300"), {})
        n_questions = int(pool.get("n_questions", 0))
        for focal_key in resolve_focal_selector(spec.get("focal"), focal_specs):
            conditions_per_focal[focal_key] = conditions_per_focal.get(
                focal_key, 0) + len(spec.get("conditions", [])) * n_questions

    for key, spec in focal_specs.items():
        revision_calls = conditions_per_focal.get(key, 0) * replicates
        r0_calls = n_main * replicates
        # Focal models DO have fallback chains now; they are pinned, so a
        # fallback can change the route but not the model.
        add("focal", key, spec, r0_calls + revision_calls, fallbacks_allowed=True)

    for key, spec in (models_config.get("weak_agents") or {}).items():
        # honest bank: 300 questions x replicates, per weak model
        add("weak peer", key, spec, n_main * replicates, fallbacks_allowed=True)

    for tier, spec in (models_config.get("judge_cascade") or {}).items():  # noqa: B007
        # The cascade fires only when regex extraction fails (~10% in Phase 2),
        # and tiers 2 and 3 only when the tier above it also fails.
        share = {"primary": 0.10, "secondary": 0.01, "tertiary": 0.002}.get(tier, 0.01)
        total_focal = sum(r["est_calls"] for r in rows if r["role"] == "focal")
        add(f"judge ({tier})", f"judge_{tier}", spec,
            int(total_focal * share), fallbacks_allowed=True)

    verifier = models_config.get("verifier_agent") or {}
    if verifier:
        mitigation = int(experiment["pools"]["mitigation100"]["n_questions"])
        add("verifier (X6)", "verifier", verifier,
            int(mitigation * replicates * 2 * 0.15), fallbacks_allowed=True)

    return rows


def print_report(rows: List[Dict[str, Any]], models_config: Dict[str, Any]) -> None:
    providers = models_config["providers"]

    print("\n" + "=" * 78)
    print("PRIMARY ROUTING — which account pays for which call")
    print("=" * 78)
    header = f"{'role':<16}{'agent':<26}{'provider':<12}{'keys':<6}{'calls':>9}{'~$':>8}"
    print(header)
    print("-" * 78)
    for row in sorted(rows, key=lambda r: (r["role"], r["agent"])):
        if row["est_cost_usd"] is None:
            cost = "?"
        else:
            cost = f"{row['est_cost_usd']:.2f}"
            if row.get("cost_is_upper_bound"):
                cost = "<" + cost
        flag = "" if row["key_envs_present"] else "  << NO KEY"
        print(f"{row['role']:<16}{row['agent']:<26}{row['provider']:<12}"
              f"{row['n_accounts']:<6}{row['est_calls']:>9,}{cost:>8}{flag}")

    total = sum(r["est_cost_usd"] or 0 for r in rows)
    bounded = any(r.get("cost_is_upper_bound") for r in rows)
    print("-" * 78)
    print(f"{'TOTAL (upper bound)' if bounded else 'TOTAL':<60}{total:>16.2f}")
    if bounded:
        print("  '<' marks a route priced at the OpenRouter rate for the same")
        print("  model because the provider publishes no simple rate card; the")
        print("  real cost on that route is lower.")

    print("\n" + "=" * 78)
    print("ACCOUNTS — one row per billing account the run will touch")
    print("=" * 78)
    by_provider: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        entry = by_provider.setdefault(
            row["provider"],
            {"calls": 0, "cost": 0.0, "agents": [], "role": "primary"})
        entry["calls"] += row["est_calls"]
        entry["cost"] += row["est_cost_usd"] or 0
        entry["agents"].append(row["agent"])
        entry["role"] = "primary"
        # A fallback provider bills only when the primary is unavailable, but
        # it is still an account the run can touch — so it belongs in the list.
        for fallback in row["fallbacks"]:
            fb_entry = by_provider.setdefault(
                fallback,
                {"calls": 0, "cost": 0.0, "agents": [], "role": "fallback only"})
            if fb_entry["role"] == "fallback only":
                fb_entry["agents"].append(f"{row['agent']} (fallback)")

    for provider, entry in sorted(by_provider.items(),
                                  key=lambda kv: -kv[1]["cost"]):
        provider_spec = providers.get(provider, {})
        declared = provider_spec.get("api_key_envs", [])
        present = _present_keys(provider_spec)
        print(f"\n{provider.upper()}   ~{entry['calls']:,} calls   ~${entry['cost']:.2f}")
        base = os.environ.get(provider_spec.get("base_url_env", ""), "") \
            or provider_spec.get("base_url_default", "")
        print(f"  base URL : {base}")
        for name in declared:
            value = os.environ.get(name, "")
            state = "SET  " if value else "unset"
            print(f"  {state} {name:<30} {_mask(value)}")
        if not present:
            print("  !! no key present — agents on this provider cannot be built")
        elif len(present) > 1:
            print(f"  round-robin across {len(present)} accounts; fund EACH — "
                  f"traffic splits about evenly")
        else:
            print("  single account carries all of this provider's traffic")
        print(f"  carries  : {', '.join(sorted(set(entry['agents'])))}")

    print("\n" + "=" * 78)
    print("FOCAL CALL SAFETY")
    print("=" * 78)
    for row in rows:
        if row["role"] != "focal":
            continue
        chain = " -> ".join([row["provider"]] + row["fallbacks"])
        print(f"  {row['agent']:<26} pin '{row['pinned_prefix']}'")
        print(f"  {'':<26} {chain}")
    print("\n  A focal fallback changes the ROUTE, never the MODEL: every link's")
    print("  served model is checked against the pin, and a link answering with")
    print("  anything else is rejected and skipped (src/pinned_fallback.py). If")
    print("  no link serves the pinned model the unit fails rather than")
    print("  returning a wrong-model answer.")


def probe(models_config: Dict[str, Any]) -> None:
    """One tiny call per distinct provider/model, to confirm keys and snapshots."""
    from src.agent_wrappers.openai_compatible_agent import build_agent_from_config

    providers = models_config["providers"]
    targets = []
    for key, spec in (models_config.get("focal_agents") or {}).items():
        targets.append(("focal", key, spec))
    verifier = models_config.get("verifier_agent") or {}
    if verifier:
        targets.append(("verifier", "verifier", verifier))

    print("\n" + "=" * 78)
    print("LIVE PROBE — one ~20-token call each")
    print("=" * 78)
    print(f"{'agent':<26}{'provider':<12}{'served model':<34}{'pin'}")
    print("-" * 78)

    for role, key, spec in targets:
        provider = spec.get("provider")
        if not _present_keys(providers.get(provider, {})):
            print(f"{key:<26}{provider:<12}{'(no key)':<34}SKIP")
            continue
        try:
            agent = build_agent_from_config(
                agent_name=key, provider_key=provider,
                model_slug=spec["model_slug"], providers_config=providers,
                max_retries=1, retry_backoff_seconds=[2], timeout_seconds=60,
            )
            response = agent.generate_response(
                system_prompt="Reply with one word.",
                user_prompt="Say OK.",
                temperature=0.0,
                maximum_output_tokens=5,
                request_metadata={"stage": "routing_probe"},
            )
            served = (response.model_name_returned_by_provider or "?").strip()
            pin = spec.get("expected_served_prefix", "")
            ok = "OK" if (not pin or served.lower().startswith(pin.lower())) \
                else "MISMATCH"
            print(f"{key:<26}{provider:<12}{served:<34}{ok}")
        except Exception as exc:
            print(f"{key:<26}{provider:<12}{'ERROR':<34}{str(exc)[:30]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true",
                        help="make one tiny call per focal provider to confirm "
                             "the key works and record the served model string")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    _load_env()

    import yaml

    from src.agents import load_models_config

    models_config = load_models_config(PROJECT_ROOT)
    with open(PROJECT_ROOT / "config" / "experiment.yaml") as handle:
        experiment = yaml.safe_load(handle)

    rows = build_rows(models_config, experiment)
    print_report(rows, models_config)
    if args.probe:
        probe(models_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
