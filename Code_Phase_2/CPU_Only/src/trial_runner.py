"""
trial_runner.py — Orchestrates individual trial execution with checkpointing.

Manages (question, condition, trial) execution, async API calls,
checkpoint-based resumability, and live progress monitoring.
"""

import os
import json
import uuid
import random
import signal
import asyncio
import logging
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .debate_protocol import run_round0, run_round1_standard, run_round1_solo_reanswer
from .confidence_weighted_protocol import run_round1_confidence_weighted
from .agent_wrappers.judge_agent import JudgeCascade

logger = logging.getLogger("platos_ship.trial_runner")

# Global flag for graceful shutdown
_SHUTDOWN_REQUESTED = False


def _signal_handler(signum, frame):
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    logger.warning("Shutdown requested (SIGINT). Finishing current trial and flushing checkpoint...")
    # Notify via SMS — import lazily to avoid circular imports
    try:
        from TextBelt import send_sms
        send_sms("\u26a0\ufe0f Plato's Ship: interrupted by SIGINT / Ctrl-C")
    except Exception:
        pass


class TrialRunner:
    """
    Orchestrates the execution of all (question, condition, trial) triplets.
    Supports checkpointing, resumability, and live monitoring.
    """

    def __init__(
        self,
        project_root: Path,
        smart_agents: Dict[str, Any],
        dumb_agents: Dict[str, Any],
        judge_cascade: JudgeCascade,
        personas_df: pd.DataFrame,
        questions_df: pd.DataFrame,
        dry_run: bool = False,
        heterogeneous_agents: Optional[Dict[str, Any]] = None,
        correct_personas_df: Optional[pd.DataFrame] = None,
        confidence_personas_df: Optional[pd.DataFrame] = None,
    ):
        self.project_root = project_root
        self.smart_agents = smart_agents
        self.dumb_agents = dumb_agents
        self.judge_cascade = judge_cascade
        self.personas_df = personas_df
        self.questions_df = questions_df
        self.dry_run = dry_run
        # Phase 2: heterogeneous-smart agents (E8), correct-anchored personas (E7),
        # and the SEPARATE confidence-bearing pool (E3, keeps main pool Phase-1-symmetric)
        self.heterogeneous_agents = heterogeneous_agents or {}
        self.correct_personas_df = correct_personas_df
        self.confidence_personas_df = confidence_personas_df

        with open(project_root / "config" / "experiment.yaml") as f:
            self.exp_config = yaml.safe_load(f)
        with open(project_root / "config" / "paths.yaml") as f:
            self.paths = yaml.safe_load(f)

        self.seed = self.exp_config["random_seed"]
        self.trial_rows = []
        self.final_answer_rows = []
        self._completed_set = set()
        self._flush_counter = 0

        # Stats
        self._total_trials = 0
        self._api_failures = 0
        self._parse_failures = 0
        self._flip_count = 0

        # Register signal handler for graceful shutdown
        try:
            signal.signal(signal.SIGINT, _signal_handler)
        except (OSError, ValueError):
            pass  # May not work in all environments

    def _resolve_path(self, key: str) -> Path:
        p = Path(self.paths[key])
        if not p.is_absolute():
            p = self.project_root / p
        return p

    def _load_checkpoint(self):
        """Load completed trials from checkpoint."""
        cp_path = self._resolve_path("completed_trials_checkpoint_file")
        if cp_path.exists():
            cp_df = pd.read_parquet(str(cp_path))
            for _, row in cp_df.iterrows():
                key = (
                    row["question_identifier"],
                    row["condition_identifier"],
                    int(row["trial_replication_index"]),
                    row["focal_smart_agent_name"],
                )
                self._completed_set.add(key)
            logger.info(f"Loaded {len(self._completed_set)} completed trials from checkpoint")

    def _save_checkpoint(self):
        """Flush checkpoint and trial log."""
        cp_path = self._resolve_path("completed_trials_checkpoint_file")
        cp_path.parent.mkdir(parents=True, exist_ok=True)

        if self._completed_set:
            records = [
                {
                    "question_identifier": t[0],
                    "condition_identifier": t[1],
                    "trial_replication_index": t[2],
                    "focal_smart_agent_name": t[3],
                }
                for t in self._completed_set
            ]
            pd.DataFrame(records).to_parquet(str(cp_path), index=False)

        # Append to trial log
        if self.trial_rows:
            tl_path = self._resolve_path("trial_log_file")
            tl_path.parent.mkdir(parents=True, exist_ok=True)
            new_df = pd.DataFrame(self.trial_rows)
            if tl_path.exists():
                existing = pd.read_parquet(str(tl_path))
                combined = pd.concat([existing, new_df], ignore_index=True)
            else:
                combined = new_df
            combined.to_parquet(str(tl_path), index=False)
            self.trial_rows = []

        # Append to final answers
        if self.final_answer_rows:
            fa_path = self._resolve_path("final_answers_file")
            fa_path.parent.mkdir(parents=True, exist_ok=True)
            new_df = pd.DataFrame(self.final_answer_rows)
            if fa_path.exists():
                existing = pd.read_parquet(str(fa_path))
                combined = pd.concat([existing, new_df], ignore_index=True)
            else:
                combined = new_df
            combined.to_parquet(str(fa_path), index=False)
            self.final_answer_rows = []

    def _get_cond_config(self, condition_id: str) -> Dict[str, Any]:
        """Look up a condition config across main / phase2 / mitigation blocks."""
        for block in ("main_conditions", "phase2_conditions", "mitigation_condition"):
            cfg = self.exp_config.get(block, {}).get(condition_id)
            if cfg:
                return cfg
        return {}

    def _resolve_agent(self, agent_name: str):
        """Resolve a focal/peer agent by name from any registry."""
        if agent_name in self.smart_agents:
            return self.smart_agents[agent_name]
        if agent_name in self.heterogeneous_agents:
            return self.heterogeneous_agents[agent_name]
        if agent_name in self.dumb_agents:
            return self.dumb_agents[agent_name]
        raise KeyError(
            f"Agent '{agent_name}' not found in smart/heterogeneous/dumb registries."
        )

    def _build_agents_for_condition(
        self,
        condition_id: str,
        question: Dict,
        trial_index: int,
        focal_agent_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Build the agent list for a condition, honouring peer_mode
        (anchored / honest / split / none) and heterogeneous-smart debate.
        """
        cond_config = self._get_cond_config(condition_id)
        peer_mode = cond_config.get("peer_mode", "anchored")

        agents = []
        rng = random.Random(self.seed + trial_index + hash(question["question_identifier"]) % 10000)

        # ── Smart agents ──
        smart_count = cond_config.get("smart_agent_count", 1)
        agent_names = ["Agent_Alpha", "Agent_Beta", "Agent_Gamma"]
        het_keys = cond_config.get("heterogeneous_smart_agent_keys")

        for i in range(smart_count):
            if het_keys:
                # E8: three DISTINCT strong models (focal is the first)
                key = het_keys[i % len(het_keys)]
                smart_agent = self._resolve_agent(key)
                model_name = getattr(smart_agent, "model_name", key)
            else:
                smart_agent = self._resolve_agent(focal_agent_name)
                model_name = getattr(smart_agent, "model_name", focal_agent_name)
            agents.append({
                "identifier": agent_names[i],
                "role": "smart_focal" if i == 0 else "smart_nonfocal",
                "agent": smart_agent,
                "model_name": model_name,
                "provider": getattr(smart_agent, "provider", "unknown"),
                "temperature": 0.7,
                "max_tokens": 600,
            })

        # ── Weak (dumb) agents ──
        dumb_count = cond_config.get("dumb_agent_count", 0)
        if dumb_count > 0:
            dumb_agent_keys = list(self.dumb_agents.keys())
            if dumb_count == 1:
                chosen_models = [rng.choice(dumb_agent_keys)]
            else:
                chosen_models = dumb_agent_keys[:2]

            for j, key in enumerate(chosen_models):
                dumb_agent = self.dumb_agents[key]

                # Decide anchoring for this peer slot given peer_mode
                if peer_mode == "honest":
                    persona_text, anchor = None, "honest"
                elif peer_mode == "split":
                    # slot 0 = wrong-anchored, slot 1 = correct-anchored
                    if j == 0:
                        persona_text = self._get_persona(question["question_identifier"], trial_index, j, rng, pool="wrong")
                        anchor = "wrong"
                    else:
                        persona_text = self._get_persona(question["question_identifier"], trial_index, j, rng, pool="correct")
                        anchor = "correct"
                else:  # "anchored" (Phase 1 default). persona_pool picks the source pool.
                    pool = cond_config.get("persona_pool", "wrong")  # "wrong" or "confidence"
                    persona_text = self._get_persona(question["question_identifier"], trial_index, j, rng, pool=pool)
                    anchor = "wrong"

                agents.append({
                    "identifier": f"Agent_Dumb_{j+1}",
                    "role": "dumb",
                    "agent": dumb_agent,
                    "model_name": getattr(dumb_agent, "model_name", key),
                    "provider": getattr(dumb_agent, "provider", "unknown"),
                    "persona_text": persona_text,
                    "persona_anchor_mode": anchor,
                    "temperature": 0.9,
                    "max_tokens": 350,
                })

        return agents

    def _get_persona(
        self, question_id: str, trial_index: int, dumb_index: int, rng, pool: str = "wrong",
    ) -> Optional[str]:
        """
        Get a validated persona variant for a question.

        pool="wrong"      -> the wrong-anchored pool (self.personas_df)
        pool="correct"    -> the correct-anchored pool (self.correct_personas_df, E7)
        pool="confidence" -> the confidence-bearing wrong pool (self.confidence_personas_df, E3)
        """
        source = self.personas_df
        if pool == "correct":
            if self.correct_personas_df is None:
                logger.warning("Split condition needs correct_personas_df but none was provided.")
                return None
            source = self.correct_personas_df
        elif pool == "confidence":
            if self.confidence_personas_df is None:
                logger.warning("C5R needs confidence_personas_df but none was provided; using main pool.")
            else:
                source = self.confidence_personas_df

        q_personas = source[
            (source["question_identifier"] == question_id) &
            (source["validation_pass_status"] == "passed")
        ]
        if q_personas.empty:
            q_personas = source[source["question_identifier"] == question_id]
        if q_personas.empty:
            return None

        variant_idx = (trial_index + dumb_index) % len(q_personas)
        return q_personas.iloc[variant_idx]["generated_persona_text"]

    def _run_single_trial(
        self,
        question: Dict,
        condition_id: str,
        trial_index: int,
        focal_agent_name: str,
    ) -> bool:
        """Execute a single trial. Returns True if successful."""
        global _SHUTDOWN_REQUESTED
        if _SHUTDOWN_REQUESTED:
            return False

        trial_uuid = str(uuid.uuid4())
        trial_seed = self.seed + trial_index * 1000 + int(hashlib.md5(question["question_identifier"].encode()).hexdigest(), 16) % 10000
        timestamp = datetime.datetime.utcnow().isoformat()

        cond_config = self._get_cond_config(condition_id)
        aggregation_rule = cond_config.get("aggregation_rule", "none")

        correct_answer = str(question["correct_answer"]).strip()
        agents = self._build_agents_for_condition(condition_id, question, trial_index, focal_agent_name)

        try:
            # Round 0
            r0_responses = run_round0(agents, question, self.judge_cascade)

            # Round 1 (skip for C1 solo)
            r1_responses = []
            filtered_out_count = None
            if aggregation_rule == "standard_debate":
                ordering_seed = trial_seed + 999
                r1_responses = run_round1_standard(
                    agents, r0_responses, question, self.judge_cascade, ordering_seed
                )
            elif aggregation_rule == "solo_reanswer":
                # C1R (E1): focal re-answers with no peer messages
                r1_responses = run_round1_solo_reanswer(
                    agents, r0_responses, question, self.judge_cascade
                )
            elif aggregation_rule == "confidence_weighted":
                ordering_seed = trial_seed + 999
                focal_info = [a for a in agents if a["role"] == "smart_focal"][0]
                dumb_infos = [a for a in agents if a["role"] == "dumb"]
                conf_threshold = cond_config.get("confidence_threshold_for_filtering", 60)
                r1_responses, filtered_out_count = run_round1_confidence_weighted(
                    focal_info, dumb_infos, r0_responses, question,
                    self.judge_cascade, ordering_seed, conf_threshold
                )

            # Build trial log rows
            for r in r0_responses:
                answer_correct = str(r["extracted_final_answer"]).strip().upper() == correct_answer.upper()
                persona_id = None
                for a in agents:
                    if a["identifier"] == r["agent_identifier"] and a["role"] == "dumb":
                        persona_id = f"{question['question_identifier']}_persona_{trial_index % 5}"

                self.trial_rows.append({
                    "trial_universal_unique_identifier": trial_uuid,
                    "question_identifier": question["question_identifier"],
                    "condition_identifier": condition_id,
                    "trial_replication_index": trial_index,
                    "focal_smart_agent_name": focal_agent_name,
                    "responding_agent_identifier": r["agent_identifier"],
                    "responding_agent_role": r["role"],
                    "responding_agent_model_name": r["model_name"],
                    "responding_agent_provider": r["provider"],
                    "debate_round_index": 0,
                    "aggregation_rule_applied": aggregation_rule,
                    "peer_messages_seen_in_context": None,
                    "peer_messages_filtered_out_count": None,
                    "peer_messages_ordering_seed": None,
                    "injected_dumb_persona_identifier": persona_id,
                    "raw_response_text": r["raw_response_text"],
                    "extracted_final_answer": r["extracted_final_answer"],
                    "extracted_self_reported_confidence_integer": r.get("extracted_self_reported_confidence_integer"),
                    "confidence_parse_status": r.get("confidence_parse_status", "missing_line"),
                    "answer_extraction_method": r["answer_extraction_method"],
                    "extracted_answer_matches_ground_truth": answer_correct,
                    "total_input_tokens": r["total_input_tokens"],
                    "total_output_tokens": r["total_output_tokens"],
                    "wall_clock_latency_seconds": r["wall_clock_latency_seconds"],
                    "error_status": r["error_status"],
                    "retry_attempts_used": r["retry_attempts_used"],
                    "timestamp_utc": timestamp,
                    "random_seed_used_for_this_trial": trial_seed,
                })

            for r in r1_responses:
                answer_correct = str(r["extracted_final_answer"]).strip().upper() == correct_answer.upper()
                self.trial_rows.append({
                    "trial_universal_unique_identifier": trial_uuid,
                    "question_identifier": question["question_identifier"],
                    "condition_identifier": condition_id,
                    "trial_replication_index": trial_index,
                    "focal_smart_agent_name": focal_agent_name,
                    "responding_agent_identifier": r["agent_identifier"],
                    "responding_agent_role": r["role"],
                    "responding_agent_model_name": r["model_name"],
                    "responding_agent_provider": r["provider"],
                    "debate_round_index": 1,
                    "aggregation_rule_applied": aggregation_rule,
                    "peer_messages_seen_in_context": r.get("peer_messages_seen_in_context"),
                    "peer_messages_filtered_out_count": r.get("peer_messages_filtered_out_count"),
                    "peer_messages_ordering_seed": r.get("peer_messages_ordering_seed"),
                    "injected_dumb_persona_identifier": None,
                    "raw_response_text": r["raw_response_text"],
                    "extracted_final_answer": r["extracted_final_answer"],
                    "extracted_self_reported_confidence_integer": r.get("extracted_self_reported_confidence_integer"),
                    "confidence_parse_status": r.get("confidence_parse_status", "missing_line"),
                    "answer_extraction_method": r["answer_extraction_method"],
                    "extracted_answer_matches_ground_truth": answer_correct,
                    "total_input_tokens": r["total_input_tokens"],
                    "total_output_tokens": r["total_output_tokens"],
                    "wall_clock_latency_seconds": r["wall_clock_latency_seconds"],
                    "error_status": r["error_status"],
                    "retry_attempts_used": r["retry_attempts_used"],
                    "timestamp_utc": timestamp,
                    "random_seed_used_for_this_trial": trial_seed,
                })

            # Build final answer row (focal agent only)
            focal_r0 = [r for r in r0_responses if r["role"] == "smart_focal"]
            focal_r1 = [r for r in r1_responses if r["role"] == "smart_focal"]

            if focal_r0:
                r0 = focal_r0[0]
                r0_correct = str(r0["extracted_final_answer"]).strip().upper() == correct_answer.upper()
                r1_answer = r0["extracted_final_answer"]
                r1_correct = r0_correct
                r1_conf = r0.get("extracted_self_reported_confidence_integer")

                if focal_r1:
                    r1 = focal_r1[0]
                    r1_answer = r1["extracted_final_answer"]
                    r1_correct = str(r1_answer).strip().upper() == correct_answer.upper()
                    r1_conf = r1.get("extracted_self_reported_confidence_integer")

                # Determine dumb peer consensus
                dumb_r0 = [r for r in r0_responses if r["role"] == "dumb"]
                dumb_count = len(dumb_r0)
                if dumb_count == 0:
                    consensus = "not_applicable"
                else:
                    dumb_correct = [
                        str(r["extracted_final_answer"]).strip().upper() == correct_answer.upper()
                        for r in dumb_r0
                    ]
                    if all(not c for c in dumb_correct):
                        consensus = "unanimous_wrong"
                    elif all(c for c in dumb_correct):
                        consensus = "unanimous_correct"
                    else:
                        consensus = "split"

                self.final_answer_rows.append({
                    "question_identifier": question["question_identifier"],
                    "condition_identifier": condition_id,
                    "trial_replication_index": trial_index,
                    "focal_smart_agent_name": focal_agent_name,
                    "aggregation_rule_applied": aggregation_rule,
                    "round_zero_independent_answer": r0["extracted_final_answer"],
                    "round_zero_answer_was_correct": r0_correct,
                    "round_zero_focal_self_reported_confidence_integer": r0.get("extracted_self_reported_confidence_integer"),
                    "round_one_post_debate_answer": r1_answer,
                    "round_one_answer_was_correct": r1_correct,
                    "round_one_focal_self_reported_confidence_integer": r1_conf,
                    "focal_agent_flipped_correct_to_incorrect": r0_correct and not r1_correct,
                    "focal_agent_flipped_incorrect_to_correct": not r0_correct and r1_correct,
                    "dumb_peer_consensus_status": consensus,
                    "condition_dumb_agent_count": dumb_count,
                    "c5_count_of_peer_messages_filtered_out": filtered_out_count,
                })

            # Update tracking
            self._total_trials += 1
            key = (question["question_identifier"], condition_id, trial_index, focal_agent_name)
            self._completed_set.add(key)

            # Periodic checkpoint
            self._flush_counter += 1
            if self._flush_counter >= 50:
                self._save_checkpoint()
                self._flush_counter = 0

            return True

        except Exception as e:
            logger.error(f"Trial failed: {condition_id}/{question['question_identifier']}/{trial_index}: {e}",
                        exc_info=True)
            self._api_failures += 1
            return False

    def run_conditions(
        self,
        conditions: List[str],
        focal_agent_name: str,
        question_filter: Optional[pd.DataFrame] = None,
        trials_per_question: Optional[int] = None,
        stage_name: str = "main",
    ) -> int:
        """
        Run all trials for specified conditions.
        Returns total trials completed.
        """
        self._load_checkpoint()

        questions = question_filter if question_filter is not None else self.questions_df

        if trials_per_question is None:
            trials_per_question = self.exp_config["trials_per_question_main_conditions"]
        if self.dry_run:
            trials_per_question = 1

        total_planned = len(questions) * len(conditions) * trials_per_question
        logger.info(
            f"Stage '{stage_name}': {len(questions)} questions × "
            f"{len(conditions)} conditions × {trials_per_question} trials = "
            f"{total_planned} planned trials"
        )

        completed_before = len(self._completed_set)

        with tqdm(total=total_planned, desc=f"Stage: {stage_name}") as pbar:
            for _, q_row in questions.iterrows():
                question = q_row.to_dict()

                for condition_id in conditions:
                    for trial_idx in range(trials_per_question):
                        key = (question["question_identifier"], condition_id, trial_idx, focal_agent_name)
                        if key in self._completed_set:
                            pbar.update(1)
                            continue

                        success = self._run_single_trial(
                            question, condition_id, trial_idx, focal_agent_name
                        )

                        pbar.update(1)

                        # Update progress bar description
                        if self._total_trials > 0 and self._total_trials % 100 == 0:
                            pbar.set_postfix({
                                "done": self._total_trials,
                                "fails": self._api_failures,
                                "stage": stage_name,
                            })

                        if _SHUTDOWN_REQUESTED:
                            logger.warning("Shutdown: flushing and exiting")
                            self._save_checkpoint()
                            return self._total_trials

        # Final flush
        self._save_checkpoint()

        completed_after = len(self._completed_set)
        new_completed = completed_after - completed_before
        logger.info(f"Stage '{stage_name}' complete: {new_completed} new trials")

        return new_completed
