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
    # Wrong anchors that state a Confidence line; used by WRconf and WRfilt.
    anchored_confidence: Dict[str, List[Dict[str, Any]]] = None  # type: ignore[assignment]

    @property
    def has_correct(self) -> bool:
        return bool(self.anchored_correct)

    @property
    def has_hedged(self) -> bool:
        return bool(self.anchored_hedged)

    @property
    def has_honest(self) -> bool:
        return bool(self.honest_bank)


def _index_personas(frame: Optional[pd.DataFrame], rule: str = "confident",
                    label: str = "") -> Dict[str, List[Dict[str, Any]]]:
    """
    Index the personas that may act as peers.

    Previously EVERY persona was indexed regardless of its validation status,
    so 14 wrong-anchored and 15 correct-anchored personas that failed
    validation served as peers. Six or seven per pool never state a final
    answer at all; the rest failed only on punctuation ("90." vs "90") and are
    valid. `is_usable_persona` re-validates with the corrected rules, keeping
    the valid ones and dropping the rest. Every question keeps at least four
    usable personas, so no question loses its condition.
    """
    from .anchored_personas import is_usable_persona

    if frame is None or frame.empty:
        return {}
    index: Dict[str, List[Dict[str, Any]]] = {}
    dropped = 0
    for row in frame.to_dict("records"):
        if not is_usable_persona(row, rule):
            dropped += 1
            continue
        index.setdefault(row["question_identifier"], []).append(row)
    for variants in index.values():
        variants.sort(key=lambda r: int(r.get("persona_variant_index", 0)))
    if dropped:
        logger.info("Pool %s: %d persona(s) excluded as unusable (%s rule).",
                    label, dropped, rule)
    return index


def _merge_disjoint(*indices: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Combine persona indices whose questions must not overlap."""
    merged: Dict[str, List[Dict[str, Any]]] = {}
    for index in indices:
        clash = set(merged) & set(index)
        if clash:
            raise ValueError(
                f"persona pools overlap on {len(clash)} question id(s), e.g. "
                f"{sorted(clash)[:3]}; merging them would silently mix pools")
        merged.update(index)
    return merged


def _index_honest(frame: Optional[pd.DataFrame]) -> Dict[tuple, List[Dict[str, Any]]]:
    if frame is None or frame.empty:
        return {}
    index: Dict[tuple, List[Dict[str, Any]]] = {}
    dropped = 0
    for row in frame.to_dict("records"):
        # A failed call or an empty message is not a peer message. Counting it
        # would let H run with a blank "peer", which is not the condition.
        if (str(row.get("error_status") or "") == "failure"
                or not str(row.get("message_text") or "").strip()):
            dropped += 1
            continue
        key = (row["question_identifier"], int(row.get("replicate", 0)))
        index.setdefault(key, []).append(row)
    for messages in index.values():
        messages.sort(key=lambda r: str(r.get("weak_model_key", "")))
    if dropped:
        logger.warning("Honest bank: %d failed or empty messages excluded; "
                       "their (question, replicate) cells skip H.", dropped)
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
        # X4's GSM-Symbolic items have their own wrong-anchored pool; without
        # it every X4 WR unit was skipped for want of a persona.
        anchored_wrong=_merge_disjoint(
            _index_personas(_read("anchored_personas_file"), "confident", "wrong"),
            _index_personas(_read("gsm_symbolic_personas_file"), "confident",
                            "gsm_symbolic_wrong")),
        anchored_correct=_index_personas(_read("correct_anchored_personas_file"),
                                         "confident", "correct"),
        # Hedged on purpose, so the no-hedging rule does not apply; its own
        # validation in hedged_personas.py decides.
        anchored_hedged=_index_personas(_read("hedged_personas_file"), "stored",
                                        "hedged"),
        honest_bank=_index_honest(_read("honest_bank_file")),
        anchored_confidence=_index_personas(_read("confidence_personas_file"),
                                            "confident", "confidence"),
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
    Choose `count` DISTINCT persona variants for one unit, or none at all.

    Every peer in a unit is a different persona. Returning fewer than `count`
    (normally an empty list) means the condition cannot be met for this
    question; the caller records why and skips the unit.

    WHAT THIS REPLACES. The previous version drew each peer independently
    WITH replacement, so with five personas per question about one WR trial
    in five (20.1% on the main pool) showed the model two supposedly
    independent peers posting identical text. The agreement controls also
    faked what they could not supply: WRagree repeated one persona, so both
    peers were the same text, and WRdiff topped up at random, so "different
    target" peers could agree. Its comment said the caller would record
    this; the caller did not. The released Phase 2 C4 had 16.4% target
    agreement, below the 20% duplicates alone would give, so it cannot have
    drawn with replacement either.

    Feasibility on the main pool: every question has at least four usable
    personas, so WR, SF, WRfilt, W, WR1 and WR4 are always satisfiable;
    WRdiff on 298 of 300 questions, WRagree on 241. Contrasts are paired by
    question, so a skipped question drops out of both arms of a comparison.
    """
    if not variants or count <= 0 or len(variants) < count:
        return []

    def target(v: Dict[str, Any]) -> str:
        return str(v.get("assigned_wrong_answer_letter_or_value", ""))

    if force_same_target:
        by_target: Dict[str, List[Dict[str, Any]]] = {}
        for variant in variants:
            by_target.setdefault(target(variant), []).append(variant)
        eligible = [by_target[t] for t in sorted(by_target)
                    if len(by_target[t]) >= count]
        if not eligible:
            return []
        group = eligible[rng.randrange(len(eligible))]
        return rng.sample(group, count)

    if force_distinct_targets:
        by_target = {}
        for variant in variants:
            by_target.setdefault(target(variant), []).append(variant)
        if len(by_target) < count:
            return []
        targets = rng.sample(sorted(by_target), count)
        return [by_target[t][rng.randrange(len(by_target[t]))] for t in targets]

    return rng.sample(variants, count)


def peer_rng(master_seed: int, question_id: str, replicate: int,
             condition: Dict[str, Any]):
    """
    The random stream that picks a unit's peers.

    Seeded on the question, the replicate and the selection constraint ONLY.
    Previously the condition name (and the focal model) were part of the
    seed, so WR, WRh, W and SF showed different personas for the same unit
    and every contrast between them mixed a content difference into the
    manipulation. Now:

      * WR, W, SF, WRh and X3's round sweep show the same personas (WRh via
        their hedged rewrites), so each contrast isolates its manipulation;
      * every focal model sees the same personas, as with the honest bank,
        so a cross-model difference is not peer sampling noise;
      * `random.sample` fills its result in draw order, so with one seed
        WR1's peer is WR's first and WR's two are WR4's first two: the dose
        conditions are nested.

    Conditions with an agreement constraint must draw differently, so the
    constraint is part of the seed.
    """
    constraint = ("same_target" if condition.get("force_same_target")
                  else "distinct_targets" if condition.get("force_distinct_targets")
                  else "any")
    return derive_rng(master_seed, "peers", question_id, replicate, constraint)


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
    rng = peer_rng(master_seed, question_id, replicate, condition)

    slot_keys = list(weak_specs)
    slot_names = [weak_specs[k].get("paper_name", k) for k in slot_keys] or ["weak"]
    diagnostics: Dict[str, Any] = {"peer_source": peer_source,
                                   "n_peers_requested": n_peers}

    if peer_source in ("none", "generic") or n_peers == 0:
        return [], diagnostics

    if peer_source == "self_samples":
        # E's peers are the model's OTHER cached answers to this question.
        # Running with fewer than designed would quietly change the treatment.
        if len(self_samples or []) < n_peers:
            diagnostics["skipped_reason"] = (
                f"self_samples_insufficient: {len(self_samples or [])} of {n_peers} "
                "other cached replicates available")
            return [], diagnostics
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
        # One failed call while building the bank would otherwise leave H
        # running with one honest peer instead of two, silently.
        if len(available) < n_peers:
            diagnostics["skipped_reason"] = (
                f"honest_bank_insufficient: {len(available)} of {n_peers} "
                f"messages for ({question_id}, r{replicate})")
            return [], diagnostics
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

    # anchored_wrong / anchored_hedged / anchored_confidence / bare_answer.
    # WRh draws from the WRONG pool exactly as WR does and then swaps in each
    # persona's hedged rewrite, so the two conditions differ in wording only.
    source_index = {"anchored_confidence": pools.anchored_confidence or {},
                    }.get(peer_source, pools.anchored_wrong)
    variants = source_index.get(question_id, [])
    if not variants:
        diagnostics["skipped_reason"] = f"{peer_source}_pool_missing_for_question"
        return [], diagnostics

    chosen = _pick_variants(
        variants, n_peers, rng,
        force_same_target=bool(condition.get("force_same_target")),
        force_distinct_targets=bool(condition.get("force_distinct_targets")),
    )
    if len(chosen) != n_peers:
        if condition.get("force_same_target"):
            why = f"no {n_peers} distinct personas share one wrong target"
        elif condition.get("force_distinct_targets"):
            why = f"fewer than {n_peers} distinct wrong targets"
        else:
            why = f"only {len(variants)} usable personas, {n_peers} needed"
        diagnostics["skipped_reason"] = f"{peer_source}_infeasible: {why}"
        return [], diagnostics
    if peer_source == "anchored_hedged":
        hedged = {str(v.get("persona_identifier")): v
                  for v in pools.anchored_hedged.get(question_id, [])}
        missing = [str(v.get("persona_identifier")) for v in chosen
                   if str(v.get("persona_identifier")) not in hedged]
        if missing:
            # Substituting another persona would make WRh differ from WR in
            # content as well as wording, which is the confound it removes.
            diagnostics["skipped_reason"] = (
                f"anchored_hedged_infeasible: no validated hedged rewrite of "
                f"{missing}")
            return [], diagnostics
        chosen = [hedged[str(v.get("persona_identifier"))] for v in chosen]
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
