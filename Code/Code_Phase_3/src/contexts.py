"""
contexts.py — Protocol B: one revision template, one varying block.

The Phase-1/2 confound (next_plan.md §0, §3) was that the focal model's
Round-1 call was *stateless*: `build_round1_prompt` showed the peers but not
the focal's own Round-0 text, and the API wrapper carries no conversation
history. The re-answer control C1R *did* show the own answer. So "C4 minus
C1R" mixed three changes at once: peers present, own answer removed, and a
different closing instruction. A "harmful flip" in C2-C4 was therefore partly
plain temperature-0.7 resampling of a fresh answer.

Protocol B fixes this. Every revision condition uses the SAME prompt:

    Question: ...
    <options>

    Your previous response:
    <cached Round-0 text>

    <CONTEXT_BLOCK>            <- the ONLY thing that varies

    Review your previous response[, and the other responses above].
    Reconsider the problem carefully and state your final answer with
    reasoning. You may keep or revise your answer. End with:
    Final answer: <your answer>
    Confidence: <integer from 0 to 100>

with identical temperature and max_output_tokens across conditions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .seeding import derive_rng

logger = logging.getLogger("platos_ship3.contexts")

SYSTEM_PROMPT = "You are a knowledgeable respondent in a group reasoning task."

# Round-0 instruction: identical to Phase 1/2 so cached initial answers are
# comparable with the released Protocol-A logs.
ROUND0_INSTRUCTION = (
    "Provide your reasoning in 2 to 3 sentences. "
    "End your response with these two lines, and nothing after them:\n"
    "Final answer: <your answer>\n"
    "Confidence: <integer from 0 to 100 representing how confident you are "
    "in your final answer>"
)

# The revision instruction. Two variants that differ ONLY in the clause naming
# the peer block, so a no-peer condition does not carry a dangling reference.
REVISION_INSTRUCTION_NO_PEERS = (
    "Review your previous response. Reconsider the problem carefully and "
    "state your final answer with reasoning. You may keep or revise your "
    "answer. End with:\n"
    "Final answer: <your answer>\n"
    "Confidence: <integer from 0 to 100 representing how confident you are "
    "in your final answer>"
)
REVISION_INSTRUCTION_WITH_PEERS = (
    "Review your previous response and the other responses above. Reconsider "
    "the problem carefully and state your final answer with reasoning. You "
    "may keep or revise your answer. End with:\n"
    "Final answer: <your answer>\n"
    "Confidence: <integer from 0 to 100 representing how confident you are "
    "in your final answer>"
)

# Condition G: a challenge with no content. Deliberately states no alternative
# answer and gives no reasoning, so any G-vs-R difference is "being told you
# may be wrong" and any WR-vs-G difference needs the message content.
GENERIC_CHALLENGE = (
    "Another participant believes your answer may be wrong but gave no details."
)


@dataclass
class PeerMessage:
    """One peer message plus the provenance needed to audit it later."""

    display_name: str                 # e.g. "Agent_Alpha" or "Candidate 1"
    text: str
    final_answer: Optional[str] = None
    confidence: Optional[int] = None
    # ── provenance (next_plan.md §2.3) ──────────────────────────────────
    message_generator_model: str = ""   # who actually WROTE this text
    nominal_peer_slot_model: str = ""   # the model the slot is labelled with
    served_model: str = ""              # what the provider returned, if live
    peer_source: str = ""               # anchored_wrong / honest / ...
    persona_identifier: Optional[str] = None
    anchor_mode: Optional[str] = None   # wrong / correct / honest / hedged
    assigned_target: Optional[str] = None  # the wrong answer this peer argues

    def to_row(self) -> Dict[str, Any]:
        return {
            "peer_display_name": self.display_name,
            "peer_text": self.text,
            "peer_final_answer": self.final_answer,
            "peer_confidence": self.confidence,
            "message_generator_model": self.message_generator_model,
            "nominal_peer_slot_model": self.nominal_peer_slot_model,
            "served_model": self.served_model,
            "peer_source": self.peer_source,
            "persona_identifier": self.persona_identifier,
            "anchor_mode": self.anchor_mode,
            "assigned_target": self.assigned_target,
        }


def format_options_block(answer_options: Any) -> str:
    """Render an MMLU-Pro style option list, or '' for numeric items."""
    if answer_options is None:
        return ""
    if not isinstance(answer_options, (str, list)):   # pandas NaN guard
        return ""
    if isinstance(answer_options, str):
        try:
            answer_options = json.loads(answer_options)
        except (ValueError, TypeError):
            return ""
    if not answer_options:
        return ""
    lines = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(answer_options)]
    return "\n\n" + "\n".join(lines)


def build_round0_prompt(question_text: str, answer_options: Any = None) -> Tuple[str, str]:
    """Stage 0: the independent initial answer. Cached and reused everywhere."""
    options_block = format_options_block(answer_options)
    user = f"Question: {question_text}{options_block}\n\n{ROUND0_INSTRUCTION}"
    return SYSTEM_PROMPT, user


# ──────────────────────────────────────────────────────────────────────────
# CONTEXT_BLOCK construction
# ──────────────────────────────────────────────────────────────────────────

def _attributed_block(peers: Sequence[PeerMessage], framing: str) -> str:
    """
    Render peer messages either as attributed peer turns (the default) or as
    unattributed candidate solutions (condition SF).

    SF holds the semantic content fixed and removes only the social framing,
    so an SF-vs-WR difference is a source-framing effect and nothing else.
    """
    if not peers:
        return ""
    if framing == "anonymous_candidate":
        body = "\n\n".join(
            f"Candidate solution {i + 1}: {p.text}" for i, p in enumerate(peers)
        )
        return f"Other candidate solutions to this question:\n\n{body}"
    body = "\n\n".join(f"{p.display_name} said: {p.text}" for p in peers)
    return f"Other agents' responses:\n\n{body}"


def build_context_block(
    peers: Sequence[PeerMessage],
    peer_source: str,
    source_framing: str = "peer_attributed",
) -> str:
    """
    Build the one varying section of the revision prompt.

    peer_source:
      none         -> "" (condition R)
      generic      -> the content-free challenge sentence (condition G)
      bare_answer  -> "<name> said: Final answer: X" lines only (condition W)
      anything else-> full peer messages, attributed or anonymised
    """
    if peer_source == "none":
        return ""
    if peer_source == "generic":
        return GENERIC_CHALLENGE
    if peer_source == "bare_answer":
        if not peers:
            return ""
        body = "\n".join(
            f"{p.display_name} said: Final answer: {p.final_answer}" for p in peers
        )
        return f"Other agents' responses:\n\n{body}"
    return _attributed_block(peers, source_framing)


def build_revision_prompt(
    question_text: str,
    answer_options: Any,
    own_previous_text: str,
    peers: Sequence[PeerMessage],
    peer_source: str,
    source_framing: str = "peer_attributed",
) -> Tuple[str, str]:
    """
    The single revision template used by EVERY condition.

    `own_previous_text` is the cached Round-0 text (round 1) or the focal's
    own previous-round text (rounds 2+). It is never omitted — that is the
    whole point of Protocol B.
    """
    options_block = format_options_block(answer_options)
    context_block = build_context_block(peers, peer_source, source_framing)

    has_peer_block = bool(context_block) and peer_source != "generic"
    instruction = (
        REVISION_INSTRUCTION_WITH_PEERS if has_peer_block
        else REVISION_INSTRUCTION_NO_PEERS
    )

    parts = [f"Question: {question_text}{options_block}",
             f"Your previous response:\n\n{own_previous_text}"]
    if context_block:
        parts.append(context_block)
    parts.append(instruction)

    return SYSTEM_PROMPT, "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# Peer selection
# ──────────────────────────────────────────────────────────────────────────

def order_peers(
    peers: List[PeerMessage], master_seed: int, *seed_parts: Any
) -> List[PeerMessage]:
    """Shuffle peer order deterministically (stable across processes)."""
    ordered = list(peers)
    derive_rng(master_seed, "peer_order", *seed_parts).shuffle(ordered)
    return ordered


def apply_confidence_filter(
    peers: Sequence[PeerMessage],
    threshold: int,
    unparseable_counts_as: str = "dropped",
) -> Tuple[List[PeerMessage], Dict[str, Any]]:
    """
    The deployed retain-high-confidence rule.

    Retains a peer whose stated confidence is >= threshold. A peer whose
    confidence cannot be parsed is dropped when `unparseable_counts_as ==
    "dropped"` (what the deployed rule actually does) or retained otherwise.

    Returns (kept_peers, diagnostics). Diagnostics feed the retention-gap
    report in analysis/gate.py, which needs the per-message retain decision
    together with the peer's correctness.
    """
    kept, decisions = [], []
    for peer in peers:
        if peer.confidence is None:
            retained = unparseable_counts_as != "dropped"
            reason = "unparseable_confidence"
        else:
            retained = peer.confidence >= threshold
            reason = "above_threshold" if retained else "below_threshold"
        decisions.append(
            {
                "peer_display_name": peer.display_name,
                "confidence": peer.confidence,
                "retained": retained,
                "reason": reason,
                "persona_identifier": peer.persona_identifier,
                "peer_final_answer": peer.final_answer,
            }
        )
        if retained:
            kept.append(peer)

    return kept, {
        "n_peers_before_filter": len(peers),
        "n_peers_retained": len(kept),
        "n_peers_dropped": len(peers) - len(kept),
        "filter_decisions": decisions,
    }


# When the filter removes every peer the focal must still be given a coherent
# prompt. Phase 1 substituted this literal text; keep it identical so the two
# runs stay comparable.
ALL_PEERS_FILTERED_TEXT = (
    "(No peer responses met the confidence threshold; respond based on your "
    "own reasoning.)"
)


def filtered_placeholder_peer() -> PeerMessage:
    """A stand-in 'message' used when the filter empties the peer block."""
    return PeerMessage(
        display_name="(system)",
        text=ALL_PEERS_FILTERED_TEXT,
        peer_source="filtered_empty",
        message_generator_model="n/a",
        nominal_peer_slot_model="n/a",
    )
