"""
judge_agent.py — 3-tier judge cascade for answer extraction.

Cascade: Gemini (primary, 4 keys RR) → Mistral (secondary) → DeepSeek (tertiary, 2 keys RR)
No retry within any tier — only fallback to the next tier.
Designed to be used only when regex extraction fails.
"""

import os
import re
import time
import logging
from typing import Optional, Dict, Any, Tuple

from .base_agent import AgentResponse, RoundRobinKeyManager

logger = logging.getLogger("platos_ship.agents.judge")
api_failure_logger = logging.getLogger("platos_ship.api_failures")

# ─── Regex patterns for robust answer extraction ───
# These are designed to minimize judge usage
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
    """
    Attempt to extract the final answer using regex patterns.
    Returns the answer string or None if no pattern matches.
    """
    if not raw_text:
        return None

    # Try each pattern, prioritizing later lines (answer is usually at the end)
    lines = raw_text.strip().split("\n")

    # First pass: look for "Final answer:" pattern (most reliable)
    for line in reversed(lines):
        line_stripped = line.strip()
        for pattern in FINAL_ANSWER_PATTERNS[:2]:
            match = re.search(pattern, line_stripped)
            if match:
                return match.group(1).strip().replace(",", "")

    # Second pass: try all other patterns
    for line in reversed(lines):
        line_stripped = line.strip()
        for pattern in FINAL_ANSWER_PATTERNS[2:]:
            match = re.search(pattern, line_stripped, re.MULTILINE)
            if match:
                return match.group(1).strip().replace(",", "")

    return None


def extract_confidence_regex(raw_text: str) -> Tuple[Optional[int], str]:
    """
    Extract confidence integer from text.
    Returns (confidence_int_or_None, parse_status).
    """
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
    3-tier judge cascade for answer extraction from debate transcripts.

    Tier 1: Gemini (4 keys round-robin) — 1 retry on transient errors
    Tier 2: Mistral (1 key)              — 1 retry on transient errors
    Tier 3: DeepSeek (2 keys round-robin)

    Falls through to the next tier only after retries are exhausted.
    """

    JUDGE_SYSTEM_PROMPT = (
        "You extract a single final answer from a model's response. "
        "Output only the answer, nothing else."
    )

    def __init__(self):
        self._gemini_keys = self._load_keys(
            ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"],
            "Gemini"
        )
        self._mistral_key = os.getenv("MISTRAL_API_KEY", "").strip()
        self._deepseek_keys = self._load_keys(
            ["DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2"], "DeepSeek"
        )

        self._gemini_model = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
        self._mistral_model = os.getenv("MISTRAL_MODEL_NAME", "mistral-small-latest")
        self._deepseek_model = os.getenv("DEEPSEEK_JUDGE_MODEL_NAME", "deepseek-chat")
        self._deepseek_base_url = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1")

        # Round-robin managers
        if self._gemini_keys:
            self._gemini_rr = RoundRobinKeyManager(self._gemini_keys, "Gemini-Judge")
        else:
            self._gemini_rr = None

        if self._deepseek_keys:
            self._deepseek_rr = RoundRobinKeyManager(self._deepseek_keys, "DeepSeek-Judge")
        else:
            self._deepseek_rr = None

        # Stats tracking
        self._tier_usage = {"gemini": 0, "mistral": 0, "deepseek": 0, "all_failed": 0}
        self._total_calls = 0

        logger.info(
            f"JudgeCascade initialized: Gemini({len(self._gemini_keys)} keys) → "
            f"Mistral({'1 key' if self._mistral_key else 'no key'}) → "
            f"DeepSeek({len(self._deepseek_keys)} keys)"
        )

    @staticmethod
    def _load_keys(env_vars: list, provider: str) -> list:
        keys = []
        for var in env_vars:
            key = os.getenv(var, "").strip()
            if key:
                keys.append(key)
        return keys

    def _build_user_prompt(
        self, question_text: str, answer_options: str, raw_text: str
    ) -> str:
        return (
            f"The question was: {question_text}\n\n"
            f"The valid answer options were: {answer_options}\n\n"
            f"The model's response was:\n{raw_text}\n\n"
            "Output the single answer the model committed to, in the exact format "
            "the question expects (a single capital letter for multiple choice, or "
            "a number for math). If the response is genuinely ambiguous, output "
            "the literal token UNPARSEABLE."
        )

    def _try_gemini(self, user_prompt: str) -> Optional[str]:
        """Tier 1: Gemini via google-generativeai.  1 retry on transient errors."""
        if not self._gemini_rr:
            return None
        for attempt in range(2):  # 1 initial + 1 retry
            try:
                import google.generativeai as genai

                api_key = self._gemini_rr.get_next_key()
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(self._gemini_model)
                response = model.generate_content(
                    f"{self.JUDGE_SYSTEM_PROMPT}\n\n{user_prompt}",
                    generation_config=genai.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=50,
                    ),
                )
                text = response.text.strip() if response.text else None
                if text and text != "UNPARSEABLE":
                    self._tier_usage["gemini"] += 1
                    return text
                return text  # Could be UNPARSEABLE
            except Exception as e:
                api_failure_logger.warning(
                    f"Gemini judge attempt {attempt + 1} failed: "
                    f"{type(e).__name__}: {str(e)[:200]}"
                )
                if attempt == 0:
                    time.sleep(2)
        return None

    def _try_mistral(self, user_prompt: str) -> Optional[str]:
        """Tier 2: Mistral via mistralai SDK.  1 retry on transient errors."""
        if not self._mistral_key:
            return None
        for attempt in range(2):  # 1 initial + 1 retry
            try:
                from mistralai import Mistral

                client = Mistral(api_key=self._mistral_key)
                response = client.chat.complete(
                    model=self._mistral_model,
                    messages=[
                        {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=50,
                )
                text = response.choices[0].message.content.strip() if response.choices else None
                if text and text != "UNPARSEABLE":
                    self._tier_usage["mistral"] += 1
                    return text
                return text
            except Exception as e:
                api_failure_logger.warning(
                    f"Mistral judge attempt {attempt + 1} failed: "
                    f"{type(e).__name__}: {str(e)[:200]}"
                )
                if attempt == 0:
                    time.sleep(2)
        return None

    def _try_deepseek(self, user_prompt: str) -> Optional[str]:
        """Tier 3: DeepSeek via openai-compatible client."""
        if not self._deepseek_rr:
            return None
        try:
            from openai import OpenAI

            api_key = self._deepseek_rr.get_next_key()
            client = OpenAI(api_key=api_key, base_url=self._deepseek_base_url, timeout=60)
            response = client.chat.completions.create(
                model=self._deepseek_model,
                messages=[
                    {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=50,
            )
            text = response.choices[0].message.content.strip() if response.choices else None
            if text and text != "UNPARSEABLE":
                self._tier_usage["deepseek"] += 1
                return text
            return text
        except Exception as e:
            api_failure_logger.warning(f"DeepSeek judge failed: {type(e).__name__}: {str(e)[:200]}")
            return None

    def extract_answer(
        self,
        question_text: str,
        answer_options: str,
        raw_text: str,
    ) -> Tuple[str, str]:
        """
        Extract answer using judge cascade.
        Returns (answer, method) where method is one of:
            'judge_gemini', 'judge_mistral', 'judge_deepseek', 'parse_failure'
        """
        self._total_calls += 1
        user_prompt = self._build_user_prompt(question_text, answer_options, raw_text)

        # Tier 1: Gemini
        result = self._try_gemini(user_prompt)
        if result and result != "UNPARSEABLE":
            return result, "judge_gemini"

        # Tier 2: Mistral
        result = self._try_mistral(user_prompt)
        if result and result != "UNPARSEABLE":
            return result, "judge_mistral"

        # Tier 3: DeepSeek
        result = self._try_deepseek(user_prompt)
        if result and result != "UNPARSEABLE":
            return result, "judge_deepseek"

        self._tier_usage["all_failed"] += 1
        return "UNPARSEABLE", "parse_failure"

    @property
    def usage_stats(self) -> Dict[str, Any]:
        return {
            "total_judge_calls": self._total_calls,
            "tier_usage": dict(self._tier_usage),
        }
