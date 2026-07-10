"""
judge_agent.py — 3-tier judge cascade for answer extraction (Phase 2, all-paid).

Cascade (all via OpenAI-compatible PAID endpoints — no free tier):
  Tier 1: gemini-2.0-flash via OpenRouter (paid)
  Tier 2: mistral-small via Mistral La Plateforme (paid)
  Tier 3: deepseek-chat via DeepSeek direct (paid)

Providers/models are read from config/models.yaml `judge_cascade:` and the
`providers:` registry, so any tier can be re-pointed to a cheaper source by
editing config. No retry within a tier — only fallback to the next tier.
Fires only when regex extraction of the final-answer line fails.
"""

import os
import re
import time
import logging
from typing import Optional, Dict, Any, Tuple, List

from .base_agent import AgentResponse
from .openai_compatible_agent import OpenAICompatibleAgent, build_agent_from_config

logger = logging.getLogger("platos_ship.agents.judge")
api_failure_logger = logging.getLogger("platos_ship.api_failures")

# ─── Regex patterns for robust answer extraction ───
# These are designed to minimize judge usage.
FINAL_ANSWER_PATTERNS = [
    # Standard: "Final answer: X"
    r"[Ff]inal\s+[Aa]nswer\s*:\s*([A-J])\b",
    r"[Ff]inal\s+[Aa]nswer\s*:\s*(-?[\d,]+\.?\d*)",
    # Variations: "The answer is X"
    r"[Tt]he\s+answer\s+is\s*:?\s*([A-J])\b",
    r"[Tt]he\s+answer\s+is\s*:?\s*(-?[\d,]+\.?\d*)",
    # Bracketed: "[X]" or "(X)" at end
    r"\[([A-J])\]\s*$",
    r"\(([A-J])\)\s*$",
    # Bold/emphasis: **X** at end
    r"\*\*([A-J])\*\*\s*$",
    # Just a letter on its own line
    r"^\s*([A-J])\s*$",
]

CONFIDENCE_PATTERN = r"[Cc]onfidence\s*:\s*(\d+)"


def extract_answer_regex(raw_text: str) -> Optional[str]:
    """Extract the final answer using regex patterns. None if no match."""
    if not raw_text:
        return None

    lines = raw_text.strip().split("\n")

    # First pass: "Final answer:" pattern (most reliable), from the end.
    for line in reversed(lines):
        line_stripped = line.strip()
        for pattern in FINAL_ANSWER_PATTERNS[:2]:
            match = re.search(pattern, line_stripped)
            if match:
                return match.group(1).strip().replace(",", "")

    # Second pass: all other patterns.
    for line in reversed(lines):
        line_stripped = line.strip()
        for pattern in FINAL_ANSWER_PATTERNS[2:]:
            match = re.search(pattern, line_stripped, re.MULTILINE)
            if match:
                return match.group(1).strip().replace(",", "")

    return None


def extract_confidence_regex(raw_text: str) -> Tuple[Optional[int], str]:
    """Extract confidence integer. Returns (confidence_or_None, parse_status)."""
    if not raw_text:
        return None, "missing_line"

    match = re.search(CONFIDENCE_PATTERN, raw_text)
    if not match:
        return None, "missing_line"

    try:
        value = int(match.group(1))
        if value < 0:
            return 0, "out_of_range_clamped"
        elif value > 100:
            return 100, "out_of_range_clamped"
        return value, "success"
    except (ValueError, TypeError):
        return None, "non_integer"


class JudgeCascade:
    """
    3-tier judge cascade, every tier a PAID OpenAI-compatible endpoint.

    Built from config/models.yaml. Falls through to the next tier only when a
    tier returns UNPARSEABLE or errors after its single retry.
    """

    JUDGE_SYSTEM_PROMPT = (
        "You extract a single final answer from a model's response. "
        "Output only the answer, nothing else."
    )

    def __init__(self, models_config: Dict[str, Any]):
        providers = models_config["providers"]
        jc = models_config["judge_cascade"]

        self._tiers: List[Dict[str, Any]] = []
        for tier_name in ("primary", "secondary", "tertiary"):
            spec = jc.get(tier_name)
            if not spec:
                continue
            try:
                agent = build_agent_from_config(
                    agent_name=f"judge_{tier_name}",
                    provider_key=spec["provider"],
                    model_slug=spec["model_slug"],
                    providers_config=providers,
                    max_retries=1,  # judge: 1 retry inside tier, then fall through
                    timeout_seconds=spec.get("request_timeout_seconds", 60),
                )
                self._tiers.append({
                    "name": tier_name,
                    "label": spec["model_slug"],
                    "agent": agent,
                    "max_tokens": spec.get("max_output_tokens", 50),
                })
            except Exception as e:
                logger.warning(
                    f"Judge tier '{tier_name}' ({spec.get('provider')}/{spec.get('model_slug')}) "
                    f"unavailable: {e}"
                )

        if not self._tiers:
            raise ValueError("No judge tiers could be initialised. Check providers/keys in .env.")

        self._tier_usage = {t["name"]: 0 for t in self._tiers}
        self._tier_usage["all_failed"] = 0
        self._total_calls = 0

        logger.info(
            "JudgeCascade (paid) initialised: "
            + " -> ".join(f"{t['name']}({t['label']})" for t in self._tiers)
        )

    def _build_user_prompt(self, question_text: str, answer_options: str, raw_text: str) -> str:
        return (
            f"The question was: {question_text}\n\n"
            f"The valid answer options were: {answer_options}\n\n"
            f"The model's response was:\n{raw_text}\n\n"
            "Output the single answer the model committed to, in the exact format "
            "the question expects (a single capital letter for multiple choice, or "
            "a number for math). If the response is genuinely ambiguous, output "
            "the literal token UNPARSEABLE."
        )

    def extract_answer(
        self, question_text: str, answer_options: str, raw_text: str,
    ) -> Tuple[str, str]:
        """
        Extract answer through the paid cascade.
        Returns (answer, method) where method is 'judge_<tier>' or 'parse_failure'.
        """
        self._total_calls += 1
        user_prompt = self._build_user_prompt(question_text, answer_options, raw_text)

        for tier in self._tiers:
            try:
                resp = tier["agent"].generate_response(
                    system_prompt=self.JUDGE_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=0.0,
                    maximum_output_tokens=tier["max_tokens"],
                )
                text = (resp.raw_text_output or "").strip()
                if text and text != "UNPARSEABLE" and resp.error_status != "failure":
                    self._tier_usage[tier["name"]] += 1
                    return text, f"judge_{tier['name']}"
            except Exception as e:
                api_failure_logger.warning(
                    f"Judge tier {tier['name']} ({tier['label']}) failed: "
                    f"{type(e).__name__}: {str(e)[:200]}"
                )

        self._tier_usage["all_failed"] += 1
        return "UNPARSEABLE", "parse_failure"

    @property
    def usage_stats(self) -> Dict[str, Any]:
        return {
            "total_judge_calls": self._total_calls,
            "tier_usage": dict(self._tier_usage),
        }
