"""Agent wrappers package."""

from .base_agent import BaseAgent, AgentResponse, RoundRobinKeyManager
from .deepseek_agent import DeepSeekAgent
from .openrouter_agent import OpenRouterAgent
from .local_huggingface_agent import LocalHuggingFaceAgent
from .judge_agent import JudgeCascade, extract_answer_regex, extract_confidence_regex

__all__ = [
    "BaseAgent", "AgentResponse", "RoundRobinKeyManager",
    "DeepSeekAgent", "OpenRouterAgent", "LocalHuggingFaceAgent",
    "JudgeCascade", "extract_answer_regex", "extract_confidence_regex",
]
