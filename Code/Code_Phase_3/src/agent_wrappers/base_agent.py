"""
base_agent.py — Abstract base class for all agent wrappers.

Provides:
- AgentResponse dataclass for uniform response objects
- RoundRobinKeyManager for cycling API keys
- BaseAgent abstract class with generate_response() interface
"""

import time
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger("platos_ship.agents")


@dataclass
class AgentResponse:
    """Uniform response object returned by every agent wrapper."""
    raw_text_output: str = ""
    wall_clock_latency_seconds: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    model_name_returned_by_provider: str = ""
    error_status: str = "success"  # success, api_error_recovered, parse_error_recovered, failure
    retry_attempts_used: int = 0
    judge_tier_used: Optional[str] = None  # For judge responses: gemini, mistral, deepseek


class RoundRobinKeyManager:
    """
    Thread-safe round-robin key manager for API key cycling.
    
    Cycles through a list of API keys in order:
    key1 -> key2 -> ... -> keyN -> key1 -> ...
    """
    
    def __init__(self, keys: List[str], provider_name: str = "unknown"):
        if not keys:
            raise ValueError(f"No API keys provided for {provider_name}")
        self._keys = list(keys)
        self._index = 0
        self._lock = threading.Lock()
        self._provider_name = provider_name
        self._total_uses = {i: 0 for i in range(len(keys))}
        logger.info(
            f"RoundRobinKeyManager initialized for {provider_name} with {len(keys)} keys"
        )
    
    def get_next_key(self) -> str:
        """Return the next key in round-robin order (thread-safe)."""
        with self._lock:
            key = self._keys[self._index]
            key_index = self._index
            self._total_uses[key_index] += 1
            self._index = (self._index + 1) % len(self._keys)
        # Log key usage with masked key for security
        masked = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
        logger.debug(
            f"{self._provider_name} using key index {key_index} ({masked}), "
            f"total uses: {self._total_uses[key_index]}"
        )
        return key
    
    @property
    def key_count(self) -> int:
        return len(self._keys)
    
    @property
    def usage_stats(self) -> Dict[int, int]:
        with self._lock:
            return dict(self._total_uses)


class BaseAgent(ABC):
    """
    Abstract base class for all agent wrappers.
    
    Every agent must implement generate_response() with this exact signature.
    """
    
    def __init__(self, agent_name: str, provider: str):
        self.agent_name = agent_name
        self.provider = provider
        self._call_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._lock = threading.Lock()
    
    @abstractmethod
    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        maximum_output_tokens: int,
        request_metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """
        Generate a response from the agent.
        
        Args:
            system_prompt: System message for the model.
            user_prompt: User message (question + context).
            temperature: Sampling temperature.
            maximum_output_tokens: Max tokens to generate.
            request_metadata: Optional metadata dict (question_id, trial_index, etc.)
        
        Returns:
            AgentResponse with all fields populated.
        """
        pass
    
    def _track_usage(self, response: AgentResponse):
        """Thread-safe usage tracking."""
        with self._lock:
            self._call_count += 1
            self._total_input_tokens += response.total_input_tokens
            self._total_output_tokens += response.total_output_tokens
    
    @property
    def usage_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "agent_name": self.agent_name,
                "provider": self.provider,
                "total_calls": self._call_count,
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
            }
