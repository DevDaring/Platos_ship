"""
peer_pools.py — assemble the peer messages for one revision unit.

Every peer message carries its provenance (next_plan.md §2.3). In Phase 1/2 an
anchored peer's text was replayed from a pool that Llama-3.1-8B generated, but
it was logged under whichever slot it filled — so a message in the "Gemma 3 4B"
slot was never written by Gemma. Nothing Gemma-specific can be inferred from
those conditions, and the paper must say so. Here `message_generator_model` and
`nominal_peer_slot_model` are separate columns and both are written out.

Also implements the controls the reviewers asked for:
  * dose        — n_peers 1 / 2 / 4 (X3)
  * agreement   — force_same_target vs force_distinct_targets (X3)
  * framing     — peer attribution vs anonymous candidate (SF, X2)
  * split       — one wrong-anchored + one correct-anchored peer (CR)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .contexts import PeerMessage
from .extraction import extract_answer_regex, extract_confidence
from .seeding import derive_rng

logger = logging.getLogger("platos_ship3.peer_pools")

PEER_DISPLAY_NAMES = ["Agent_Beta", "Agent_Gamma", "Agent_Delta", "Agent_Epsilon"]


@dataclass
class PersonaPools:
    """Every pre-generated message pool, indexed by question for fast lookup."""

    anchored_wrong: Dict[str, List[Dict[str, Any]]]
    anchored_correct: Dict[str, List[Dict[str, Any]]]
    anchored_hedged: Dict[str, List[Dict[str, Any]]]
    honest_bank: Dict[tuple, List[Dict[str, Any]]]     # (question_id, replicate)

    @property
    def has_correct(self) -> bool:
        return bool(self.anchored_correct)

    @property
    def has_hedged(self) -> bool:
        return bool(self.anchored_hedged)

    @property
    def has_honest(self) -> bool:
        return bool(self.honest_bank)


def _index_personas(frame: Optional[pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
    if frame is None or frame.empty:
        return {}
    index: Dict[str, List[Dict[str, Any]]] = {}
    for row in frame.to_dict("records"):
        index.setdefault(row["question_identifier"], []).append(row)
    for variants in index.values():
        variants.sort(key=lambda r: int(r.get("persona_variant_index", 0)))
    return index


def _index_honest(frame: Optional[pd.DataFrame]) -> Dict[tuple, List[Dict[str, Any]]]:
    if frame is None or frame.empty:
        return {}
    index: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in frame.to_dict("records"):
        key = (row["question_identifier"], int(row.get("replicate", 0)))
        index.setdefault(key, []).append(row)
    for messages in index.values():
        messages.sort(key=lambda r: str(r.get("weak_model_key", "")))
    return index


def load_pools(paths: Dict[str, str], project_root: Path) -> PersonaPools:
    """Load whichever pools exist; missing optional pools degrade gracefully."""

    def _read(key: str) -> Optional[pd.DataFrame]:
        raw = paths.get(key)
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = project_root / raw
        if not path.exists():
            logger.info("Pool '%s' absent at %s (conditions needing it are skipped).",
                        key, path)
            return None
        return pd.read_parquet(path)

    pools = PersonaPools(
        anchored_wrong=_index_personas(_read("anchored_personas_file")),
        anchored_correct=_index_personas(_read("correct_anchored_personas_file")),
        anchored_hedged=_index_personas(_read("hedged_personas_file")),
        honest_bank=_index_honest(_read("honest_bank_file")),
    )
    logger.info(
        "Pools loaded: wrong=%d q, correct=%d q, hedged=%d q, honest=%d (q,rep)",
        len(pools.anchored_wrong), len(pools.anchored_correct),
        len(pools.anchored_hedged), len(pools.honest_bank),
    )
    return pools


def _persona_to_peer(
    persona: Dict[str, Any],
    display_name: str,
    slot_model: str,
    peer_source: str,
    anchor_mode: str,
) -> PeerMessage:
    text = persona.get("generated_persona_text", "") or ""
    confidence, _ = extract_confidence(text)
    return PeerMessage(
        display_name=display_name,
        text=text,
        final_answer=extract_answer_regex(text)
        or persona.get("assigned_wrong_answer_letter_or_value"),
        confidence=confidence,
        message_generator_model=persona.get("generator_model_name", "") or "",
        nominal_peer_slot_model=slot_model,
        served_model=persona.get("generator_model_name", "") or "",
        peer_source=peer_source,
        persona_identifier=persona.get("persona_identifier"),
        anchor_mode=anchor_mode,
        assigned_target=persona.get("assigned_wrong_answer_letter_or_value"),
    )


def _pick_variants(
    variants: List[Dict[str, Any]],
    count: int,
    rng,
    force_same_target: bool = False,
    force_distinct_targets: bool = False,
) -> List[Dict[str, Any]]:
    """
    Choose `count` persona variants for one unit.

    Agreement control (X3): the anchored pool samples a wrong target per
    variant, so two peers usually name *different* wrong answers (16.4% agree
    in the released C4). `force_same_target` selects variants sharing one
    target; `force_distinct_targets` requires different ones. Holding all peers
    wrong while varying only agreement is the clean unanimity contrast — unlike
    C4split, which removes a wrong peer AND adds correct evidence at once.
    """
    if not variants:
        return []

    def target(v: Dict[str, Any]) -> str:
        return str(v.get("assigned_wrong_answer_letter_or_value", ""))

    if force_same_target:
        by_target: Dict[str, List[Dict[str, Any]]] = {}
        for variant in variants:
            by_target.setdefault(target(variant), []).append(variant)
        eligible = [g for g in by_target.values() if len(g) >= count]
        if eligible:
            group = eligible[rng.randrange(len(eligible))]
            return [group[i % len(group)] for i in range(count)]
        # Fall back: repeat one variant so both peers state the same answer.
        chosen = variants[rng.randrange(len(variants))]
        return [chosen] * count

    if force_distinct_targets:
        seen, picked = set(), []
        for variant in sorted(variants, key=lambda v: rng.random()):
            if target(variant) in seen:
                continue
            seen.add(target(variant))
            picked.append(variant)
            if len(picked) == count:
                return picked
        # Not enough distinct targets; top up and let the caller record it.
        while len(picked) < count and variants:
            picked.append(variants[rng.randrange(len(variants))])
        return picked

    picked = []
    for _ in range(count):
        picked.append(variants[rng.randrange(len(variants))])
    return picked


def build_peers(
    condition: Dict[str, Any],
    question: Dict[str, Any],
    replicate: int,
    pools: PersonaPools,
    weak_specs: Dict[str, Any],
    master_seed: int,
    focal_key: str,
    self_samples: Optional[List[Dict[str, Any]]] = None,
) -> tuple[List[PeerMessage], Dict[str, Any]]:
    """
    Assemble the peer list for one revision unit.

    Returns (peers, diagnostics). Diagnostics record what was actually
    available, so a condition that silently degraded is visible in the logs
    rather than being mistaken for a null result.
    """
    peer_source = condition.get("peer_source", "none")
    n_peers = int(condition.get("n_peers", 0))
    question_id = question["question_identifier"]
    rng = derive_rng(master_seed, "peers", focal_key, question_id, replicate,
                     condition.get("_name", peer_source))

    slot_keys = list(weak_specs)
    slot_names = [weak_specs[k].get("paper_name", k) for k in slot_keys] or ["weak"]
    diagnostics: Dict[str, Any] = {"peer_source": peer_source,
                                   "n_peers_requested": n_peers}

    if peer_source in ("none", "generic") or n_peers == 0:
        return [], diagnostics

    if peer_source == "self_samples":
        peers = []
        for i, sample in enumerate((self_samples or [])[:n_peers]):
            peers.append(
                PeerMessage(
                    display_name=PEER_DISPLAY_NAMES[i % len(PEER_DISPLAY_NAMES)],
                    text=sample.get("raw_response_text", ""),
                    final_answer=sample.get("extracted_answer"),
                    confidence=sample.get("extracted_confidence"),
                    message_generator_model=sample.get("focal_served_model", focal_key),
                    nominal_peer_slot_model=sample.get("focal_served_model", focal_key),
                    served_model=sample.get("focal_served_model", ""),
                    peer_source="self_samples",
                    anchor_mode="self",
                )
            )
        diagnostics["n_peers_built"] = len(peers)
        return peers, diagnostics

    if peer_source == "honest":
        available = pools.honest_bank.get((question_id, replicate), [])
        peers = []
        for i, message in enumerate(available[:n_peers]):
            text = message.get("message_text", "") or ""
            confidence, _ = extract_confidence(text)
            peers.append(
                PeerMessage(
                    display_name=PEER_DISPLAY_NAMES[i % len(PEER_DISPLAY_NAMES)],
                    text=text,
                    final_answer=message.get("extracted_answer"),
                    confidence=confidence,
                    message_generator_model=message.get("served_model", "")
                    or message.get("weak_model_key", ""),
                    nominal_peer_slot_model=message.get("weak_model_key", ""),
                    served_model=message.get("served_model", ""),
                    peer_source="honest",
                    anchor_mode="honest",
                )
            )
        diagnostics["n_peers_built"] = len(peers)
        diagnostics["n_honest_available"] = len(available)
        # Honest peers are correct roughly half the time; record the realised
        # composition so the observational stratification is auditable.
        diagnostics["n_honest_peers_correct"] = sum(
            1 for m in available[:n_peers] if bool(m.get("is_correct"))
        )
        return peers, diagnostics

    if peer_source == "anchored_split":
        wrong_variants = pools.anchored_wrong.get(question_id, [])
        correct_variants = pools.anchored_correct.get(question_id, [])
        if not wrong_variants or not correct_variants:
            diagnostics["skipped_reason"] = "split_pool_missing"
            return [], diagnostics
        wrong = _pick_variants(wrong_variants, 1, rng)[0]
        correct = _pick_variants(correct_variants, 1, rng)[0]
        peers = [
            _persona_to_peer(wrong, PEER_DISPLAY_NAMES[0],
                             slot_names[0], "anchored_split", "wrong"),
            _persona_to_peer(correct, PEER_DISPLAY_NAMES[1],
                             slot_names[1 % len(slot_names)], "anchored_split",
                             "correct"),
        ]
        diagnostics["n_peers_built"] = len(peers)
        return peers, diagnostics

    # anchored_wrong / anchored_hedged
    source_index = (pools.anchored_hedged if peer_source == "anchored_hedged"
                    else pools.anchored_wrong)
    variants = source_index.get(question_id, [])
    if not variants:
        diagnostics["skipped_reason"] = f"{peer_source}_pool_missing_for_question"
        return [], diagnostics

    chosen = _pick_variants(
        variants, n_peers, rng,
        force_same_target=bool(condition.get("force_same_target")),
        force_distinct_targets=bool(condition.get("force_distinct_targets")),
    )
    anchor_mode = "hedged" if peer_source == "anchored_hedged" else "wrong"
    peers = [
        _persona_to_peer(
            variant,
            PEER_DISPLAY_NAMES[i % len(PEER_DISPLAY_NAMES)],
            slot_names[i % len(slot_names)],
            peer_source,
            anchor_mode,
        )
        for i, variant in enumerate(chosen)
    ]

    targets = {p.assigned_target for p in peers}
    diagnostics.update(
        n_peers_built=len(peers),
        n_distinct_targets=len(targets),
        peers_agree_on_target=len(targets) == 1,
    )
    return peers, diagnostics


def primary_wrong_target(peers: List[PeerMessage]) -> Optional[str]:
    """
    The wrong answer the peers argue for, used for target-adoption analysis.

    When peers disagree this returns the modal target; the analysis also keeps
    the full target set, because "revised to a peer's answer" and "revised to
    some other wrong answer" are different events.
    """
    targets = [p.assigned_target for p in peers
               if p.assigned_target and p.anchor_mode in ("wrong", "hedged")]
    if not targets:
        return None
    counts: Dict[str, int] = {}
    for target in targets:
        counts[target] = counts.get(target, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
