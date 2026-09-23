"""
test_pinned_fallback.py — a fallback may change the route, never the model.

This is the regression test for the Phase-2 contamination: a fallback chain
served 75 focal responses from `gpt-4.1-mini` while logging them as
GPT-4o-mini. Phase 3 keeps the chain (LinkAPI rate-limits, so availability
matters) but makes it incapable of substituting a different model.

All offline: the links are stubs, no network.
"""

from __future__ import annotations

import pytest

from src.agent_wrappers.base_agent import AgentResponse
from src.pinned_fallback import PinnedFallbackAgent


class StubAgent:
    """A link in the chain that returns a scripted response."""

    def __init__(self, provider: str, served_model: str, text: str = "ok",
                 error_status: str = "success"):
        self.provider = provider
        self.model_name = served_model
        self.served_model = served_model
        self.text = text
        self.error_status = error_status
        self.calls = 0

    def generate_response(self, system_prompt, user_prompt, temperature,
                          maximum_output_tokens, request_metadata=None):
        self.calls += 1
        return AgentResponse(
            raw_text_output=self.text,
            model_name_returned_by_provider=self.served_model,
            error_status=self.error_status,
        )


def _chain(*links) -> PinnedFallbackAgent:
    return PinnedFallbackAgent(
        agent_name="gpt4o_mini",
        links=list(links),
        expected_served_prefix="gpt-4o-mini",
        labels=[link.provider for link in links],
    )


def _call(agent):
    return agent.generate_response("sys", "user", 0.0, 10, {"unit": "t1"})


class TestPinnedFallback:

    def test_primary_serving_the_pin_is_used(self):
        primary = StubAgent("linkapi", "gpt-4o-mini-2024-07-18")
        second = StubAgent("openai_direct", "gpt-4o-mini-2024-07-18")
        agent = _chain(primary, second)
        response = _call(agent)
        assert response.model_name_returned_by_provider.startswith("gpt-4o-mini")
        assert primary.calls == 1
        assert second.calls == 0, "fallback used while the primary worked"

    def test_route_changes_when_primary_fails(self):
        primary = StubAgent("linkapi", "", text="", error_status="failure")
        second = StubAgent("openai_direct", "gpt-4o-mini-2024-07-18")
        agent = _chain(primary, second)
        response = _call(agent)
        assert response.error_status == "api_error_recovered"
        assert second.calls == 1
        assert agent.route_counts == {"openai_direct": 1}

    def test_wrong_model_is_rejected_not_returned(self):
        # THE Phase-2 regression: the fallback answered, but with another model.
        primary = StubAgent("linkapi", "", text="", error_status="failure")
        wrong = StubAgent("openai_direct", "gpt-4.1-mini-2025-04-14")
        right = StubAgent("openrouter", "gpt-4o-mini-2024-07-18")
        agent = _chain(primary, wrong, right)

        response = _call(agent)
        assert response.model_name_returned_by_provider == "gpt-4o-mini-2024-07-18"
        assert wrong.calls == 1, "the wrong-model link should have been tried"
        assert right.calls == 1, "the chain should have continued past it"
        assert agent.route_counts == {"openrouter": 1}
        assert len(agent.rejections) == 1
        assert agent.rejections[0]["served_model"] == "gpt-4.1-mini-2025-04-14"

    def test_all_links_wrong_model_is_a_failure_not_an_answer(self):
        # Better to fail loudly than to return a wrong-model answer.
        links = [StubAgent(f"p{i}", "gpt-4.1-mini-2025-04-14") for i in range(3)]
        agent = _chain(*links)
        response = _call(agent)
        assert response.error_status == "snapshot_unavailable"
        assert len(agent.rejections) == 3
        assert agent.route_counts == {}

    def test_rejection_records_context_for_the_appendix(self):
        primary = StubAgent("linkapi", "gpt-4.1-mini-2025-04-14")
        second = StubAgent("openai_direct", "gpt-4o-mini-2024-07-18")
        agent = _chain(primary, second)
        _call(agent)
        rejection = agent.rejections[0]
        assert rejection["route"] == "linkapi"
        assert rejection["expected_prefix"] == "gpt-4o-mini"
        assert rejection["unit"] == "t1", "request metadata must be carried through"

    def test_provenance_summary_is_reportable(self):
        primary = StubAgent("linkapi", "gpt-4o-mini-2024-07-18")
        agent = _chain(primary, StubAgent("openrouter", "gpt-4o-mini-2024-07-18"))
        _call(agent)
        _call(agent)
        summary = agent.provenance_summary()
        assert summary["accepted_by_route"] == {"linkapi": 2}
        assert summary["n_rejected_wrong_model"] == 0
        assert summary["chain"] == ["linkapi", "openrouter"]

    def test_empty_pin_accepts_anything(self):
        # Peer/judge agents have no pin; they must not be broken by this class.
        agent = PinnedFallbackAgent(
            agent_name="peer", links=[StubAgent("openrouter", "whatever-model")],
            expected_served_prefix=None,
        )
        response = _call(agent)
        assert response.model_name_returned_by_provider == "whatever-model"


class TestChainConfiguration:
    """The configured chain matches what the run is supposed to use."""

    @staticmethod
    def _spec():
        from pathlib import Path

        import yaml

        root = Path(__file__).resolve().parent.parent
        with open(root / "config" / "models.yaml") as handle:
            return yaml.safe_load(handle)["focal_agents"]["gpt4o_mini"]

    def test_gpt4o_mini_chain_is_linkapi_then_openai_then_openrouter(self):
        spec = self._spec()
        assert spec["provider"] == "linkapi"
        providers = [f["provider"] for f in spec["fallbacks"]]
        assert providers == ["openai_direct", "openrouter"]

    def test_first_fallback_has_no_retry(self):
        spec = self._spec()
        assert spec["fallbacks"][0]["max_retries"] == 0

    def test_chain_is_pinned(self):
        assert self._spec()["expected_served_prefix"] == "gpt-4o-mini"

    def test_openai_direct_provider_uses_the_dedicated_key(self):
        from pathlib import Path

        import yaml

        root = Path(__file__).resolve().parent.parent
        with open(root / "config" / "models.yaml") as handle:
            providers = yaml.safe_load(handle)["providers"]
        # An sk-proj-… platform key authenticates against api.openai.com and is
        # rejected (401) by OpenRouter, so it must be routed here.
        assert providers["openai_direct"]["api_key_envs"] == ["Open_AI_2009_Key"]
        assert "api.openai.com" in providers["openai_direct"]["base_url_default"]
