"""
debate_protocol.py — Standard debate protocol for conditions C1–C4.

Implements Round 0 (independent) and Round 1 (peer-exposed) with
randomized peer message ordering and answer/confidence extraction.
"""

import json
import random
import logging
from typing import Dict, List, Optional, Tuple, Any

from .agent_wrappers.judge_agent import extract_answer_regex, extract_confidence_regex, JudgeCascade

logger = logging.getLogger("platos_ship.debate_protocol")

# Universal structured-output instruction (used in all conditions C1–C5)
STRUCTURED_OUTPUT_INSTRUCTION = (
    "Provide your reasoning in 2 to 3 sentences. "
    "End your response with these two lines, and nothing after them:\n"
    "Final answer: <your answer>\n"
    "Confidence: <integer from 0 to 100 representing how confident you are in your final answer>"
)

ROUND_1_INSTRUCTION = (
    "Review the responses above. State your final answer with reasoning. "
    "You may agree or disagree. End with:\n"
    "Final answer: <your answer>\n"
    "Confidence: <integer from 0 to 100 representing how confident you are in your final answer>"
)


def build_round0_prompt(
    question_text: str,
    answer_options: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """Build Round 0 independent prompt."""
    system = "You are a knowledgeable respondent in a group reasoning task."

    options_block = ""
    if answer_options:
        if isinstance(answer_options, str):
            answer_options = json.loads(answer_options)
        options_block = "\n".join(
            f"{chr(65+i)}. {opt}" for i, opt in enumerate(answer_options)
        )
        options_block = f"\n\n{options_block}"

    user = (
        f"Question: {question_text}{options_block}\n\n"
        f"{STRUCTURED_OUTPUT_INSTRUCTION}"
    )

    return system, user


def build_round1_prompt(
    question_text: str,
    answer_options: Optional[List[str]],
    peer_responses: List[Dict[str, Any]],
    ordering_seed: int,
) -> Tuple[str, str]:
    """
    Build Round 1 peer-exposed prompt with randomized peer ordering.

    peer_responses: List of dicts with keys:
        'agent_identifier', 'response_text', 'confidence' (int or None)
    """
    system = "You are a knowledgeable respondent in a group reasoning task."

    options_block = ""
    if answer_options:
        if isinstance(answer_options, str):
            answer_options = json.loads(answer_options)
        options_block = "\n".join(
            f"{chr(65+i)}. {opt}" for i, opt in enumerate(answer_options)
        )
        options_block = f"\n\n{options_block}"

    # Randomize peer message order
    rng = random.Random(ordering_seed)
    shuffled_peers = list(peer_responses)
    rng.shuffle(shuffled_peers)

    peer_block = "\n\n".join(
        f"{p['agent_identifier']} said: {p['response_text']}"
        for p in shuffled_peers
    )

    user = (
        f"Question: {question_text}{options_block}\n\n"
        f"Other agents' responses:\n\n{peer_block}\n\n"
        f"{ROUND_1_INSTRUCTION}"
    )

    return system, user


def extract_answer_with_fallback(
    raw_text: str,
    question_text: str,
    answer_options_str: str,
    judge_cascade: JudgeCascade,
) -> Tuple[str, str]:
    """
    Extract answer: regex first, then judge cascade fallback.
    Returns (answer, extraction_method).
    """
    # Step 1: regex
    answer = extract_answer_regex(raw_text)
    if answer:
        return answer, "regex_success"

    # Step 2: judge cascade (Gemini → Mistral → DeepSeek, no retry)
    answer, method = judge_cascade.extract_answer(
        question_text=question_text,
        answer_options=answer_options_str,
        raw_text=raw_text,
    )
    return answer, method


def run_round0(
    agents: List[Dict[str, Any]],
    question: Dict,
    judge_cascade: JudgeCascade,
) -> List[Dict[str, Any]]:
    """
    Execute Round 0 (independent) for all agents in a condition.

    agents: List of dicts with keys:
        'identifier' (str), 'role' (str), 'agent' (BaseAgent instance),
        'persona_text' (str or None — for dumb agents only)
    
    Returns list of response dicts.
    """
    system, user = build_round0_prompt(
        question_text=question["question_text"],
        answer_options=question.get("answer_options"),
    )

    answer_options_str = ""
    if question.get("answer_options"):
        opts = question["answer_options"]
        if isinstance(opts, str):
            opts = json.loads(opts)
        answer_options_str = ", ".join(f"{chr(65+i)}" for i in range(len(opts)))

    responses = []
    for agent_info in agents:
        agent = agent_info["agent"]
        identifier = agent_info["identifier"]
        role = agent_info["role"]
        persona_text = agent_info.get("persona_text")

        if role == "dumb" and persona_text:
            # Dumb agents use their pre-generated persona text
            response_text = persona_text

            # Extract answer from persona text
            answer, method = extract_answer_with_fallback(
                persona_text, question["question_text"], answer_options_str, judge_cascade,
            )
            confidence, conf_status = extract_confidence_regex(persona_text)

            responses.append({
                "agent_identifier": identifier,
                "role": role,
                "model_name": agent_info.get("model_name", "unknown"),
                "provider": agent_info.get("provider", "unknown"),
                "raw_response_text": persona_text,
                "extracted_final_answer": answer,
                "extracted_self_reported_confidence_integer": confidence,
                "confidence_parse_status": conf_status,
                "answer_extraction_method": method,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "wall_clock_latency_seconds": 0.0,
                "error_status": "success",
                "retry_attempts_used": 0,
            })
        else:
            # Smart agents generate independently
            agent_response = agent.generate_response(
                system_prompt=system,
                user_prompt=user,
                temperature=agent_info.get("temperature", 0.7),
                maximum_output_tokens=agent_info.get("max_tokens", 600),
                request_metadata={"question_id": question["question_identifier"], "round": 0},
            )

            answer, method = extract_answer_with_fallback(
                agent_response.raw_text_output, question["question_text"],
                answer_options_str, judge_cascade,
            )
            confidence, conf_status = extract_confidence_regex(agent_response.raw_text_output)

            responses.append({
                "agent_identifier": identifier,
                "role": role,
                "model_name": agent_response.model_name_returned_by_provider,
                "provider": agent_info.get("provider", "unknown"),
                "raw_response_text": agent_response.raw_text_output,
                "extracted_final_answer": answer,
                "extracted_self_reported_confidence_integer": confidence,
                "confidence_parse_status": conf_status,
                "answer_extraction_method": method,
                "total_input_tokens": agent_response.total_input_tokens,
                "total_output_tokens": agent_response.total_output_tokens,
                "wall_clock_latency_seconds": agent_response.wall_clock_latency_seconds,
                "error_status": agent_response.error_status,
                "retry_attempts_used": agent_response.retry_attempts_used,
            })

    return responses


def run_round1_standard(
    agents: List[Dict[str, Any]],
    round0_responses: List[Dict[str, Any]],
    question: Dict,
    judge_cascade: JudgeCascade,
    ordering_seed: int,
) -> List[Dict[str, Any]]:
    """
    Execute Round 1 (standard debate) for conditions C2, C3, C4.
    Each agent sees all OTHER agents' Round 0 responses.
    """
    answer_options_str = ""
    if question.get("answer_options"):
        opts = question["answer_options"]
        if isinstance(opts, str):
            opts = json.loads(opts)
        answer_options_str = ", ".join(f"{chr(65+i)}" for i in range(len(opts)))

    responses = []
    for agent_info in agents:
        agent = agent_info["agent"]
        identifier = agent_info["identifier"]
        role = agent_info["role"]

        # Build peer responses (exclude self)
        peer_responses = [
            {
                "agent_identifier": r["agent_identifier"],
                "response_text": r["raw_response_text"],
                "confidence": r.get("extracted_self_reported_confidence_integer"),
            }
            for r in round0_responses
            if r["agent_identifier"] != identifier
        ]

        system, user = build_round1_prompt(
            question_text=question["question_text"],
            answer_options=question.get("answer_options"),
            peer_responses=peer_responses,
            ordering_seed=ordering_seed,
        )

        if role == "dumb":
            # Dumb agents in standard debate also generate a Round 1 response
            agent_response = agent.generate_response(
                system_prompt=system,
                user_prompt=user,
                temperature=agent_info.get("temperature", 0.9),
                maximum_output_tokens=agent_info.get("max_tokens", 350),
                request_metadata={"question_id": question["question_identifier"], "round": 1},
            )
        else:
            agent_response = agent.generate_response(
                system_prompt=system,
                user_prompt=user,
                temperature=agent_info.get("temperature", 0.7),
                maximum_output_tokens=agent_info.get("max_tokens", 600),
                request_metadata={"question_id": question["question_identifier"], "round": 1},
            )

        answer, method = extract_answer_with_fallback(
            agent_response.raw_text_output, question["question_text"],
            answer_options_str, judge_cascade,
        )
        confidence, conf_status = extract_confidence_regex(agent_response.raw_text_output)

        responses.append({
            "agent_identifier": identifier,
            "role": role,
            "model_name": agent_response.model_name_returned_by_provider,
            "provider": agent_info.get("provider", "unknown"),
            "raw_response_text": agent_response.raw_text_output,
            "extracted_final_answer": answer,
            "extracted_self_reported_confidence_integer": confidence,
            "confidence_parse_status": conf_status,
            "answer_extraction_method": method,
            "peer_messages_seen_in_context": json.dumps([
                {"agent": p["agent_identifier"], "text": p["response_text"][:200], "confidence": p["confidence"]}
                for p in peer_responses
            ]),
            "peer_messages_ordering_seed": ordering_seed,
            "total_input_tokens": agent_response.total_input_tokens,
            "total_output_tokens": agent_response.total_output_tokens,
            "wall_clock_latency_seconds": agent_response.wall_clock_latency_seconds,
            "error_status": agent_response.error_status,
            "retry_attempts_used": agent_response.retry_attempts_used,
        })

    return responses
