"""Agent wrappers package (Phase 2 — API-only, multi-provider)."""

from .base_agent import BaseAgent, AgentResponse, RoundRobinKeyManager
from .deepseek_agent import DeepSeekAgent
from .openrouter_agent import OpenRouterAgent
from .openai_compatible_agent import (
    OpenAICompatibleAgent, build_agent_from_config, resolve_provider,
)
from .judge_agent import JudgeCascade, extract_answer_regex, extract_confidence_regex

__all__ = [
    "BaseAgent", "AgentResponse", "RoundRobinKeyManager",
    "DeepSeekAgent", "OpenRouterAgent",
    "OpenAICompatibleAgent", "build_agent_from_config", "resolve_provider",
    "JudgeCascade", "extract_answer_regex", "extract_confidence_regex",
]
