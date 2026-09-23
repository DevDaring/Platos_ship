"""
Agent wrappers.

Copied verbatim from `Code_Phase_2/CPU_Only/src/agent_wrappers/` (commit of the
released artefact) so Phase 3 is a self-contained archive. Keeping the proven
round-robin/backoff/fallback behaviour identical means a Protocol-A vs
Protocol-B difference cannot be an artefact of changed request handling.
"""

from .base_agent import AgentResponse, BaseAgent, RoundRobinKeyManager
from .judge_agent import JudgeCascade, extract_answer_regex, extract_confidence_regex
from .openai_compatible_agent import (
    FallbackAgent,
    OpenAICompatibleAgent,
    build_agent_from_config,
)

__all__ = [
    "AgentResponse",
    "BaseAgent",
    "RoundRobinKeyManager",
    "JudgeCascade",
    "extract_answer_regex",
    "extract_confidence_regex",
    "FallbackAgent",
    "OpenAICompatibleAgent",
    "build_agent_from_config",
]
