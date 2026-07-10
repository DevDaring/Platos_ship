"""
openai_compatible_agent.py — Generic OpenAI-compatible chat agent (Phase 2).

Almost every provider the study touches speaks the OpenAI Chat Completions
protocol: OpenRouter, DeepSeek direct, Mistral La Plateforme, LinkAPI.ai,
nano-gpt.com, Together, Fireworks, and GCP Vertex's OpenAI-compat endpoint.
This one wrapper points at any of them via (base_url, api_keys, model_string),
so a model can be routed to whichever provider is cheapest simply by editing
config/models.yaml — no code change.

Round-robin key rotation + exponential backoff are inherited behaviour, kept
identical to the Phase 1 DeepSeek/OpenRouter wrappers for reproducibility.
"""

import os
import time
import random
import logging
from typing import Optional, Dict, Any, List

from openai import OpenAI, APIError, APITimeoutError, RateLimitError, APIConnectionError

from .base_agent import BaseAgent, AgentResponse, RoundRobinKeyManager

logger = logging.getLogger("platos_ship.agents.openai_compatible")
api_failure_logger = logging.getLogger("platos_ship.api_failures")

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class OpenAICompatibleAgent(BaseAgent):
    """
    One agent class for every OpenAI-compatible provider.

    Args:
        agent_name:   Logical name (e.g. "sweep_llama_3_1_70b").
        provider:     Provider tag used for logging/provenance (e.g. "nanogpt").
        model_name:   Provider-specific model slug (e.g. "meta-llama/llama-3.1-70b-instruct").
        api_keys:     One or more keys; rotated round-robin.
        base_url:     Provider OpenAI-compatible base URL.
        default_headers: Extra headers (OpenRouter wants HTTP-Referer / X-Title).
    """

    def __init__(
        self,
        agent_name: str,
        provider: str,
        model_name: str,
        api_keys: List[str],
        base_url: str,
        max_retries: int = 5,
        retry_backoff_seconds: Optional[list] = None,
        timeout_seconds: int = 120,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(agent_name=agent_name, provider=provider)
        if not model_name:
            raise ValueError(f"No model_name for agent '{agent_name}' (provider={provider}).")
        if not base_url:
            raise ValueError(f"No base_url for agent '{agent_name}' (provider={provider}).")
        clean_keys = [k for k in (api_keys or []) if k]
        if not clean_keys:
            raise ValueError(
                f"No API keys for agent '{agent_name}' (provider={provider}). "
                f"Check the provider's *_API_KEY_* entries in .env."
            )

        self.model_name = model_name
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff_seconds or [2, 4, 8, 16, 32]
        self.timeout_seconds = timeout_seconds
        self.default_headers = default_headers or {}
        self._key_manager = RoundRobinKeyManager(clean_keys, provider_name=provider)

        logger.info(
            f"OpenAICompatibleAgent '{agent_name}' ready: provider={provider}, "
            f"model={model_name}, keys={self._key_manager.key_count}, base_url={base_url}"
        )

    def _create_client(self) -> OpenAI:
        api_key = self._key_manager.get_next_key()
        kwargs = dict(api_key=api_key, base_url=self.base_url, timeout=self.timeout_seconds)
        if self.default_headers:
            kwargs["default_headers"] = self.default_headers
        return OpenAI(**kwargs)

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
                # Guard: providers/aggregators can transiently return a 200 with
                # null/empty choices (upstream provider error, moderation, or a
                # timeout wrapped as 200). Indexing choices[0] would crash; treat
                # this as a RETRYABLE condition so the next attempt (a rotated
                # key) can recover, exactly like an API error.
                choices = getattr(response, "choices", None)
                message = choices[0].message if choices else None
                if message is None:
                    err_payload = getattr(response, "error", None)
                    last_error = f"malformed response: empty choices (payload: {str(err_payload)[:200]})"
                    api_failure_logger.warning(
                        f"{self.provider} malformed response (attempt {attempt + 1}/{self.max_retries + 1}): "
                        f"{last_error}, model={self.model_name}, meta={metadata}"
                    )
                    if attempt < self.max_retries:
                        delay = self.retry_backoff[min(attempt, len(self.retry_backoff) - 1)]
                        time.sleep(delay + random.uniform(0, 1))
                        continue
                    break  # retries exhausted -> fall through to failure result

                elapsed = time.time() - start_time
                usage = response.usage
                result = AgentResponse(
                    raw_text_output=message.content or "",
                    wall_clock_latency_seconds=elapsed,
                    total_input_tokens=usage.prompt_tokens if usage else 0,
                    total_output_tokens=usage.completion_tokens if usage else 0,
                    model_name_returned_by_provider=response.model or self.model_name,
                    error_status="success" if attempt == 0 else "api_error_recovered",
                    retry_attempts_used=attempt,
                )
                self._track_usage(result)
                return result

            except (RateLimitError, APIError, APITimeoutError, APIConnectionError) as e:
                last_error = e
                status_code = getattr(e, "status_code", None)
                api_failure_logger.warning(
                    f"{self.provider} error (attempt {attempt + 1}/{self.max_retries + 1}): "
                    f"status={status_code}, {type(e).__name__}: {str(e)[:200]}, meta={metadata}"
                )
                if attempt < self.max_retries:
                    delay = self.retry_backoff[min(attempt, len(self.retry_backoff) - 1)]
                    time.sleep(delay + random.uniform(0, 1))
            except Exception as e:
                last_error = e
                api_failure_logger.error(
                    f"{self.provider} unexpected: {type(e).__name__}: {str(e)[:500]}, meta={metadata}"
                )
                break

        elapsed = time.time() - start_time
        result = AgentResponse(
            raw_text_output="",
            wall_clock_latency_seconds=elapsed,
            error_status="failure",
            retry_attempts_used=self.max_retries,
            model_name_returned_by_provider=self.model_name,
        )
        self._track_usage(result)
        logger.error(f"{self.provider} call failed after {self.max_retries + 1} attempts: {last_error}")
        return result

    @property
    def key_usage_stats(self) -> Dict[int, int]:
        return self._key_manager.usage_stats


class FallbackAgent(BaseAgent):
    """
    Tries a primary agent, then falls back to alternates on failure.

    Used when a model is served by a single-account provider that can rate-limit
    (e.g. gpt-4o-mini on LinkAPI). If the primary returns a failure after its own
    retries, the same request is transparently reissued on the next provider
    (e.g. nano-gpt). Result is identical either way — only the route changes.
    """

    def __init__(self, agent_name: str, primary: OpenAICompatibleAgent,
                 fallbacks: List[OpenAICompatibleAgent]):
        super().__init__(agent_name=agent_name, provider=f"{primary.provider}+fallback")
        self.primary = primary
        self.fallbacks = fallbacks or []
        self.model_name = primary.model_name  # for provenance/labelling
        self._chain = [primary] + self.fallbacks

    def generate_response(
        self, system_prompt: str, user_prompt: str, temperature: float,
        maximum_output_tokens: int, request_metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        last = None
        for i, agent in enumerate(self._chain):
            resp = agent.generate_response(
                system_prompt, user_prompt, temperature, maximum_output_tokens, request_metadata,
            )
            if resp.error_status != "failure" and (resp.raw_text_output or "").strip():
                if i > 0:
                    resp.error_status = "api_error_recovered"  # a fallback provider served it
                    logger.info(f"{self.agent_name}: primary failed, recovered via fallback #{i} ({agent.provider})")
                return resp
            last = resp
        api_failure_logger.error(
            f"{self.agent_name}: all {len(self._chain)} providers failed "
            f"({', '.join(a.provider for a in self._chain)})"
        )
        return last if last is not None else AgentResponse(error_status="failure",
                                                           model_name_returned_by_provider=self.model_name)

    @property
    def key_usage_stats(self) -> Dict[int, int]:
        return self.primary.key_usage_stats


# ─────────────────────────────────────────────────────────────────────────
# Provider registry helpers
# ─────────────────────────────────────────────────────────────────────────

# OpenRouter asks for these headers for attribution; harmless elsewhere.
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://platos-ship-debate-study.research",
    "X-Title": "Platos Ship Capability-Asymmetric Debate Study (Phase 2)",
}


def resolve_provider(provider_key: str, providers_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve a provider entry from config into concrete (base_url, keys, headers).

    providers_config[provider_key] must define:
        base_url_env:   env var holding the base URL
        api_key_envs:   list of env vars holding one or more keys
        default_headers (optional): "openrouter" | mapping | None
    """
    if provider_key not in providers_config:
        raise KeyError(
            f"Provider '{provider_key}' is not defined in config/models.yaml 'providers:'. "
            f"Known providers: {sorted(providers_config.keys())}"
        )
    entry = providers_config[provider_key]

    base_url = os.getenv(entry["base_url_env"], entry.get("base_url_default", "")).strip()
    keys = [os.getenv(v, "").strip() for v in entry.get("api_key_envs", [])]
    keys = [k for k in keys if k]

    headers_spec = entry.get("default_headers")
    if headers_spec == "openrouter":
        headers = OPENROUTER_HEADERS
    elif isinstance(headers_spec, dict):
        headers = headers_spec
    else:
        headers = None

    return {"base_url": base_url, "keys": keys, "headers": headers}


def build_agent_from_config(
    agent_name: str,
    provider_key: str,
    model_slug: str,
    providers_config: Dict[str, Any],
    max_retries: int = 5,
    retry_backoff_seconds: Optional[list] = None,
    timeout_seconds: int = 120,
) -> OpenAICompatibleAgent:
    """Factory: build an OpenAICompatibleAgent for (provider_key, model_slug)."""
    resolved = resolve_provider(provider_key, providers_config)
    return OpenAICompatibleAgent(
        agent_name=agent_name,
        provider=provider_key,
        model_name=model_slug,
        api_keys=resolved["keys"],
        base_url=resolved["base_url"],
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        timeout_seconds=timeout_seconds,
        default_headers=resolved["headers"],
    )
