"""
deepseek_agent.py — Smart agent wrapper for DeepSeek API.

Uses the openai Python client pointed at DeepSeek's base URL.
Implements round-robin key cycling across 2 keys and exponential
backoff retry on transient HTTP errors.
"""

import os
import time
import random
import logging
from typing import Optional, Dict, Any

from openai import OpenAI, APIError, APITimeoutError, RateLimitError, APIConnectionError

from .base_agent import BaseAgent, AgentResponse, RoundRobinKeyManager

logger = logging.getLogger("platos_ship.agents.deepseek")
api_failure_logger = logging.getLogger("platos_ship.api_failures")

# HTTP status codes that warrant retry
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class DeepSeekAgent(BaseAgent):
    """
    DeepSeek API agent with round-robin key management and exponential backoff.
    
    Cycles through DEEPSEEK_API_KEY_1 and DEEPSEEK_API_KEY_2 in round-robin.
    Retries on 429, 500, 502, 503, 504 with delays [2, 4, 8, 16, 32] + jitter.
    """
    
    def __init__(
        self,
        agent_name: str = "deepseek_primary",
        model_name: Optional[str] = None,
        api_keys: Optional[list] = None,
        base_url: Optional[str] = None,
        max_retries: int = 5,
        retry_backoff_seconds: Optional[list] = None,
        timeout_seconds: int = 120,
    ):
        super().__init__(agent_name=agent_name, provider="deepseek")
        
        # Resolve configuration from environment if not provided
        self.model_name = model_name or os.getenv("DEEPSEEK_PRIMARY_MODEL_NAME", "deepseek-chat")
        self.base_url = base_url or os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1")
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff_seconds or [2, 4, 8, 16, 32]
        self.timeout_seconds = timeout_seconds
        
        # Set up round-robin key manager
        if api_keys is None:
            api_keys = []
            for var in ["DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2"]:
                key = os.getenv(var)
                if key:
                    api_keys.append(key.strip())
        
        if not api_keys:
            raise ValueError("No DeepSeek API keys found. Set DEEPSEEK_API_KEY_1 and DEEPSEEK_API_KEY_2.")
        
        self._key_manager = RoundRobinKeyManager(api_keys, provider_name="DeepSeek")
        
        logger.info(
            f"DeepSeekAgent '{agent_name}' initialized: model={self.model_name}, "
            f"keys={self._key_manager.key_count}, max_retries={max_retries}"
        )
    
    def _create_client(self) -> OpenAI:
        """Create an OpenAI client with the next round-robin key."""
        api_key = self._key_manager.get_next_key()
        return OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
        )
    
    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        maximum_output_tokens: int,
        request_metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """
        Call DeepSeek API with retry logic and round-robin key cycling.
        """
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
                status_code = getattr(e, 'status_code', None)
                
                api_failure_logger.warning(
                    f"DeepSeek API error (attempt {attempt + 1}/{self.max_retries + 1}): "
                    f"status={status_code}, error={type(e).__name__}: {str(e)[:200]}, "
                    f"metadata={metadata}"
                )
                
                if attempt < self.max_retries:
                    delay = self.retry_backoff[min(attempt, len(self.retry_backoff) - 1)]
                    jitter = random.uniform(0, 1)
                    total_delay = delay + jitter
                    logger.info(f"Retrying in {total_delay:.1f}s...")
                    time.sleep(total_delay)
                    
            except Exception as e:
                last_error = e
                api_failure_logger.error(
                    f"DeepSeek unexpected error: {type(e).__name__}: {str(e)[:500]}, "
                    f"metadata={metadata}"
                )
                break
        
        # All retries exhausted
        elapsed = time.time() - start_time
        result = AgentResponse(
            raw_text_output="",
            wall_clock_latency_seconds=elapsed,
            error_status="failure",
            retry_attempts_used=self.max_retries,
            model_name_returned_by_provider=self.model_name,
        )
        self._track_usage(result)
        logger.error(
            f"DeepSeek call failed after {self.max_retries + 1} attempts: {last_error}"
        )
        return result
    
    @property
    def key_usage_stats(self) -> Dict[int, int]:
        return self._key_manager.usage_stats
