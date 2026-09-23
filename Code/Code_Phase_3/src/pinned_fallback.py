"""
pinned_fallback.py — a fallback chain that cannot change the model.

Why this exists
---------------
Phase 2's `FallbackAgent` accepts any non-empty response from any link in the
chain. When LinkAPI rate-limited, the chain failed over and 75 focal responses
came back from `gpt-4.1-mini` while still being logged as GPT-4o-mini. The
contamination was not caused by having a cheap primary; it was caused by a
fallback that was allowed to serve a *different model*.

So a focal model may have a fallback chain — for availability — provided the
chain can only ever change the ROUTE, never the MODEL. `PinnedFallbackAgent`
enforces that:

  * every link's response is checked against `expected_served_prefix`;
  * a link that serves something else is rejected and the chain moves on,
    exactly as if it had errored;
  * if no link serves the pinned model, the result is a failure with
    `error_status="snapshot_unavailable"` rather than a wrong-model answer;
  * every rejection is logged and counted, so the appendix can state how often
    it happened.

A rejected response still cost money. That is the correct trade: a wrong-model
answer costs a retracted claim.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from .agent_wrappers.base_agent import AgentResponse, BaseAgent

logger = logging.getLogger("platos_ship3.pinned_fallback")


class PinnedFallbackAgent(BaseAgent):
    """
    Try each agent in order; accept only a response from the pinned model.

    `links` is an ordered list of (agent, label) pairs. The first is the
    primary. Each agent carries its own retry policy, so a link configured
    with `max_retries=0` is tried exactly once before the chain moves on.
    """

    def __init__(
        self,
        agent_name: str,
        links: List[Any],
        expected_served_prefix: Optional[str],
        labels: Optional[List[str]] = None,
    ):
        primary = links[0]
        super().__init__(agent_name=agent_name,
                         provider=f"{primary.provider}+pinned_fallback")
        self._links = list(links)
        self._labels = labels or [getattr(a, "provider", "?") for a in links]
        self.expected_served_prefix = (expected_served_prefix or "").strip()
        self.model_name = getattr(primary, "model_name", agent_name)
        self._lock = threading.Lock()
        self._route_counts: Dict[str, int] = {}
        self._rejections: List[Dict[str, Any]] = []
        logger.info(
            "'%s' pinned chain: %s (pin=%r)",
            agent_name, " -> ".join(self._labels), self.expected_served_prefix,
        )

    def _matches_pin(self, served: str) -> bool:
        if not self.expected_served_prefix:
            return True
        return (served or "").strip().lower().startswith(
            self.expected_served_prefix.lower())

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        maximum_output_tokens: int,
        request_metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        last: Optional[AgentResponse] = None

        for index, agent in enumerate(self._links):
            label = self._labels[index]
            response = agent.generate_response(
                system_prompt, user_prompt, temperature,
                maximum_output_tokens, request_metadata,
            )
            usable = (response.error_status != "failure"
                      and (response.raw_text_output or "").strip())
            if not usable:
                last = response
                continue

            served = response.model_name_returned_by_provider or ""
            if not self._matches_pin(served):
                # The link answered, but with the wrong model. Reject it.
                with self._lock:
                    self._rejections.append({
                        "route": label,
                        "served_model": served,
                        "expected_prefix": self.expected_served_prefix,
                        **(request_metadata or {}),
                    })
                logger.warning(
                    "'%s': %s served '%s' but the pin is '%s' — rejecting and "
                    "continuing down the chain.",
                    self.agent_name, label, served, self.expected_served_prefix,
                )
                last = response
                continue

            with self._lock:
                self._route_counts[label] = self._route_counts.get(label, 0) + 1
            if index > 0:
                response.error_status = "api_error_recovered"
                logger.info("'%s': primary unavailable; served by %s (model pin holds).",
                            self.agent_name, label)
            return response

        logger.error(
            "'%s': no link in the chain (%s) served the pinned model '%s'.",
            self.agent_name, " -> ".join(self._labels), self.expected_served_prefix,
        )
        failure = last if last is not None else AgentResponse()
        failure.error_status = "snapshot_unavailable"
        return failure

    # ── provenance ────────────────────────────────────────────────────────

    @property
    def route_counts(self) -> Dict[str, int]:
        """How many accepted responses each route served."""
        with self._lock:
            return dict(self._route_counts)

    @property
    def rejections(self) -> List[Dict[str, Any]]:
        """Every response rejected for serving the wrong model."""
        with self._lock:
            return list(self._rejections)

    def provenance_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "agent": self.agent_name,
                "expected_served_prefix": self.expected_served_prefix,
                "chain": list(self._labels),
                "accepted_by_route": dict(self._route_counts),
                "n_rejected_wrong_model": len(self._rejections),
                "rejected_models": sorted(
                    {r["served_model"] for r in self._rejections}),
            }

    @property
    def key_usage_stats(self) -> Dict[int, int]:
        return getattr(self._links[0], "key_usage_stats", {})
