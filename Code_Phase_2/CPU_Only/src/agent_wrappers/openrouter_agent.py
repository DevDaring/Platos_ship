"""
openrouter_agent.py — Smart agent wrapper for OpenRouter API (GPT-4o-mini).

Uses the openai Python client with OpenRouter's base URL.
Round-robin key cycling across 2 keys with exponential backoff retry.
"""

import os
import time
import random
import logging
from typing import Optional, Dict, Any

from openai import OpenAI, APIError, APITimeoutError, RateLimitError, APIConnectionError

from .base_agent import BaseAgent, AgentResponse, RoundRobinKeyManager

logger = logging.getLogger("platos_ship.agents.openrouter")
api_failure_logger = logging.getLogger("platos_ship.api_failures")


class OpenRouterAgent(BaseAgent):
    """OpenRouter API agent with round-robin key management."""

    def __init__(
        self,
        agent_name: str = "openrouter_gpt4o_mini",
        model_name: Optional[str] = None,
        api_keys: Optional[list] = None,
        base_url: Optional[str] = None,
        max_retries: int = 5,
        retry_backoff_seconds: Optional[list] = None,
        timeout_seconds: int = 120,
    ):
        super().__init__(agent_name=agent_name, provider="openrouter")
        self.model_name = model_name or os.getenv("OPENROUTER_PRIMARY_MODEL_NAME", "openai/gpt-4o-mini")
        self.base_url = base_url or os.getenv("OPENROUTER_API_BASE_URL", "https://openrouter.ai/api/v1")
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff_seconds or [2, 4, 8, 16, 32]
        self.timeout_seconds = timeout_seconds

        if api_keys is None:
            api_keys = [os.getenv(v, "").strip() for v in ["OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2"] if os.getenv(v)]
        if not api_keys:
            raise ValueError("No OpenRouter API keys found.")
        self._key_manager = RoundRobinKeyManager(api_keys, provider_name="OpenRouter")
        logger.info(f"OpenRouterAgent initialized: model={self.model_name}, keys={self._key_manager.key_count}")

    def _create_client(self) -> OpenAI:
        api_key = self._key_manager.get_next_key()
        return OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            default_headers={
                "HTTP-Referer": "https://platos-ship-debate-study.research",
                "X-Title": "Platos Ship Capability-Asymmetric Debate Study",
            },
        )

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        maximum_output_tokens: int,
        request_metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        metadata = request_metadata or {}
        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                client = self._create_client()
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=maximum_output_tokens,
                )
                elapsed = time.time() - start_time
                result = AgentResponse(
                    raw_text_output=response.choices[0].message.content or "",
                    wall_clock_latency_seconds=elapsed,
                    total_input_tokens=response.usage.prompt_tokens if response.usage else 0,
                    total_output_tokens=response.usage.completion_tokens if response.usage else 0,
                    model_name_returned_by_provider=response.model or self.model_name,
                    error_status="success" if attempt == 0 else "api_error_recovered",
                    retry_attempts_used=attempt,
                )
                self._track_usage(result)
                return result
            except (RateLimitError, APIError, APITimeoutError, APIConnectionError) as e:
                last_error = e
                api_failure_logger.warning(
                    f"OpenRouter error (attempt {attempt+1}/{self.max_retries+1}): "
                    f"{type(e).__name__}: {str(e)[:200]}, metadata={metadata}"
                )
                if attempt < self.max_retries:
                    delay = self.retry_backoff[min(attempt, len(self.retry_backoff) - 1)]
                    time.sleep(delay + random.uniform(0, 1))
            except Exception as e:
                last_error = e
                api_failure_logger.error(f"OpenRouter unexpected: {type(e).__name__}: {str(e)[:500]}")
                break

        elapsed = time.time() - start_time
        result = AgentResponse(
            raw_text_output="", wall_clock_latency_seconds=elapsed,
            error_status="failure", retry_attempts_used=self.max_retries,
            model_name_returned_by_provider=self.model_name,
        )
        self._track_usage(result)
        return result

    @property
    def key_usage_stats(self) -> Dict[int, int]:
        return self._key_manager.usage_stats
