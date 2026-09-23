"""
judge_agent.py — 3-tier judge cascade for answer extraction (Phase 2, all-paid).

Cascade (all via OpenAI-compatible PAID endpoints — no free tier):
  Tier 1: gemini-2.5-flash  — LinkAPI -> Google direct -> OpenRouter
  Tier 2: mistral-small     — Mistral La Plateforme
  Tier 3: deepseek-v4-flash — DeepSeek direct

Providers/models are read from config/models.yaml `judge_cascade:` and the
`providers:` registry, so any tier can be re-pointed to a cheaper source by
editing config. Fires only when regex extraction of the final-answer line
fails.

PHASE 3 CHANGE (the one edit to this otherwise verbatim Phase-2 file): a tier
may declare `fallbacks:`, which route the SAME model through alternative
providers before the cascade drops to the next tier's different model. Tier 1
is Gemini via LinkAPI -> Google direct (4 keys, round-robin) -> OpenRouter.
"""

import os
import re
import time
import logging
from typing import Optional, Dict, Any, Tuple, List

from .base_agent import AgentResponse
from .openai_compatible_agent import (
    FallbackAgent,
    OpenAICompatibleAgent,
    build_agent_from_config,
)

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
        "You are an EXTRACTOR, not a solver. You are shown a question and a "
        "model's response. Your only job is to report the final answer THAT "
        "RESPONSE STATED.\n"
        "Rules:\n"
        "1. Never compute, derive, verify or correct anything. You are not "
        "being asked what the right answer is.\n"
        "2. The answer you output must appear in the response text. If you "
        "cannot point to it there, you do not have one.\n"
        "3. If the response reasons without committing, trails off, is cut "
        "off mid-sentence, or states no final answer, output exactly "
        "UNPARSEABLE.\n"
        "4. A response being wrong is not a reason to output UNPARSEABLE. "
        "Report the wrong answer it committed to.\n"
        "Output only the answer or UNPARSEABLE, nothing else."
    )

    @staticmethod
    def _option_text_for_letter(letter: str, options_text: str) -> str:
        """
        The option body a letter labels, from an "A) foo; B) bar" listing.

        Returns "" when the listing is not in that form, which makes the
        caller fall back to letter matching alone.
        """
        if not letter or not options_text:
            return ""
        # The listing ends with a trailing instruction ("-- answer with the
        # single capital letter"). Without "--" as a terminator the LAST
        # option absorbed it, so its body never matched anything in a
        # response and that option could not be grounded by its text.
        pattern = (r"(?:^|[;\n])\s*" + re.escape(letter.upper())
                   + r"\)\s*(.+?)(?=\s*(?:;|\n|--|$))")
        found = re.search(pattern, options_text, re.DOTALL)
        return found.group(1).strip() if found else ""

    @staticmethod
    def _is_grounded(answer: str, raw_text: str, options_text: str = "") -> bool:
        """
        The judge's answer must occur in the text it was asked to read.

        Prompt wording alone does not stop a competent model from solving:
        shown "What is 8 * 7?" and a response that only says "I need to think
        about this more carefully", tier 1 returned 56. Nothing in that
        response says 56. This is the hard guard, because it does not depend
        on the judge cooperating.

        Deliberately conservative. A response that writes "forty-two" while
        the judge reports "42" is treated as ungrounded and abstains. That
        loses a recoverable row, which is the safe direction: an abstention
        keeps the trial in the ambiguous set where the bounds already handle
        it, whereas a fabricated answer enters the analysis as data.
        """
        candidate = (answer or "").strip()
        text = raw_text or ""
        if not candidate:
            return False

        # An option letter is grounded EITHER by the letter appearing as its
        # own token OR by the option's text appearing in the response.
        #
        # Letter-only matching was wrong in both directions. Matching case
        # insensitively made "A" and "I" almost always match, because "a"
        # and "I" are ordinary English words, so those two letters were
        # never really checked. Meanwhile a response that said "the answer
        # is bonobos" without ever writing "A" was rejected, although the
        # judge had read it correctly rather than solved it. Comparing
        # against the option body covers that case, which is common here
        # precisely because these are the rows whose "Final answer:" line
        # the regex could not find.
        if len(candidate) == 1 and candidate.isalpha():
            letter = re.escape(candidate.upper())
            body = JudgeCascade._option_text_for_letter(candidate, options_text)
            if body:
                trimmed = re.sub(r"\s+", " ", body).strip().lower()
                haystack = re.sub(r"\s+", " ", text).lower()
                if len(trimmed) >= 2 and trimmed in haystack:
                    return True
            # "A" and "I" are English words in their own right, so a bare
            # capital token proves nothing for those two: "I think" would
            # ground option I on any response at all. They need the letter
            # to appear where a CHOICE is being stated.
            if candidate.upper() in ("A", "I"):
                contexts = [
                    r"(?:answer|option|choice|select|pick|chose|choose|go(?:ing)? with)"
                    r"[^A-Za-z0-9]{0,15}" + letter + r"(?![A-Za-z])",
                    r"\(\s*" + letter + r"\s*\)",
                    r"\*\*\s*" + letter + r"\s*\*\*",
                    r"(?m)^\s*" + letter + r"[.):]?\s*$",
                ]
                return any(re.search(p, text, re.IGNORECASE) for p in contexts)
            return re.search(
                r"(?<![A-Za-z])" + letter + r"(?![A-Za-z])", text) is not None

        def _normalise(number: str) -> str:
            number = number.replace(",", "").replace(" ", "")
            if "." in number:
                number = number.rstrip("0").rstrip(".")
            return number or "0"

        stripped = candidate.replace(",", "").replace(" ", "")
        if re.fullmatch(r"-?\d+(?:\.\d+)?", stripped):
            target = _normalise(stripped)
            for match in re.finditer(r"-?\d[\d,]*(?:\.\d+)?", text):
                if _normalise(match.group(0)) == target:
                    return True
            return False

        return candidate.lower() in text.lower()

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
                # PHASE 3 ADDITION: a tier may declare its own `fallbacks:`.
                # Tier 1 is Gemini, routed LinkAPI -> Google direct (4 keys,
                # round-robin) -> OpenRouter. Without this the cascade would
                # drop straight to a different MODEL (Mistral) the moment one
                # Gemini route hiccuped, which wastes the cheap route and
                # changes the extractor for no reason.
                route_fallbacks = []
                for index, fb in enumerate(spec.get("fallbacks") or []):
                    try:
                        route_fallbacks.append(build_agent_from_config(
                            agent_name=f"judge_{tier_name}_fb{index + 1}",
                            provider_key=fb["provider"],
                            model_slug=fb["model_slug"],
                            providers_config=providers,
                            max_retries=int(fb.get("max_retries", 1)),
                            timeout_seconds=spec.get("request_timeout_seconds", 60),
                        ))
                    except Exception as fb_error:
                        logger.warning(
                            f"Judge tier '{tier_name}' fallback via "
                            f"{fb.get('provider')} unavailable: {fb_error}"
                        )
                if route_fallbacks:
                    agent = FallbackAgent(
                        agent_name=f"judge_{tier_name}",
                        primary=agent,
                        fallbacks=route_fallbacks,
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
                # AN ABSTENTION IS A VERDICT, NOT A TIER FAILURE.
                #
                # This loop previously escalated on UNPARSEABLE exactly as it
                # escalated on a 500, so a response that never stated an
                # answer was handed to the next tier, and that tier -- being
                # a competent model shown a question and some text -- SOLVED
                # it. Observed directly: for "I need to think about this more
                # carefully before committing to anything..." on "What is
                # 8 * 7?", tier 1 correctly abstained and tier 2 returned 56.
                # That converts a missing observation into a fabricated one,
                # and it biases towards the judge's own competence rather
                # than the focal model's behaviour.
                #
                # Escalate only when the PROVIDER failed. When a tier answers
                # and its answer is "no answer was stated", that is the
                # result.
                if resp.error_status == "failure":
                    continue
                text = (resp.raw_text_output or "").strip()
                if not text:
                    continue
                # Accept any spelling of the abstention token. Tiers have
                # been observed replying "UNPARSE", and with a strict
                # equality test that fell through to the grounding guard
                # and was recorded as a fabrication rather than as the
                # abstention it plainly is.
                if re.match(r"^[^A-Za-z0-9]*UNPARS", text, re.IGNORECASE):
                    self._tier_usage[tier["name"]] += 1
                    return "UNPARSEABLE", f"abstain_{tier['name']}"
                if not self._is_grounded(text, raw_text, answer_options):
                    # The tier answered with something the response never
                    # said, i.e. it solved the problem instead of reading
                    # it. That is a fabricated observation, so it is
                    # refused here rather than escalated: the next tier
                    # would be just as able to solve it.
                    api_failure_logger.warning(
                        "Judge tier %s returned %r, which does not occur in "
                        "the response; treating as abstention",
                        tier["name"], text[:40])
                    self._tier_usage[tier["name"]] += 1
                    return "UNPARSEABLE", f"ungrounded_{tier['name']}"
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
