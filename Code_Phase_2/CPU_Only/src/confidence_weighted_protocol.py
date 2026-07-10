"""
confidence_weighted_protocol.py — C5-specific debate protocol.

Implements confidence-weighted peer aggregation per §7.5:
- Peers with confidence < threshold are excluded from focal agent's context.
- Peers with confidence >= threshold are labelled with confidence scores.
- Empty peer block handled as solo-equivalent.
"""

import json
import random
import logging
from typing import Dict, List, Optional, Tuple, Any

from .agent_wrappers.judge_agent import extract_answer_regex, extract_confidence_regex, JudgeCascade
from .debate_protocol import build_round0_prompt, extract_answer_with_fallback

logger = logging.getLogger("platos_ship.confidence_weighted")

C5_INSTRUCTION_PREFIX = (
    "Below are responses from other agents. Each response is labelled with "
    "the agent's self-reported confidence on a scale of 0 to 100. Treat "
    "low-confidence responses with appropriate scepticism. Some responses "
    "may have been filtered out because the responding agent indicated low "
    "confidence in its own answer."
)

C5_EMPTY_PEERS_NOTE = (
    "(No peer responses met the confidence threshold; respond based on your own reasoning.)"
)

C5_FINAL_INSTRUCTION = (
    "Provide your final answer. End with:\n"
    "Final answer: <your answer>\n"
    "Confidence: <integer from 0 to 100 representing how confident you are in your final answer>"
)


def filter_peers_by_confidence(
    round0_responses: List[Dict[str, Any]],
    focal_identifier: str,
    confidence_threshold: int = 60,
) -> Tuple[List[Dict], int]:
    """
    Filter peer responses by confidence threshold for C5.
    
    Returns:
        (filtered_peers, count_filtered_out)
    """
    peers = [r for r in round0_responses if r["agent_identifier"] != focal_identifier]
    
    included = []
    filtered_out = 0
    
    for peer in peers:
        confidence = peer.get("extracted_self_reported_confidence_integer")
        
        # If confidence is None (parse failure), treat as below threshold
        if confidence is None or confidence < confidence_threshold:
            filtered_out += 1
            logger.debug(
                f"C5: Filtered out {peer['agent_identifier']} "
                f"(confidence={confidence}, threshold={confidence_threshold})"
            )
        else:
            included.append(peer)
    
    return included, filtered_out


def build_c5_round1_prompt(
    question_text: str,
    answer_options,
    filtered_peers: List[Dict[str, Any]],
    ordering_seed: int,
) -> Tuple[str, str]:
    """Build the C5 confidence-weighted Round 1 prompt for the focal smart agent."""
    system = "You are a knowledgeable respondent in a group reasoning task."

    # Guard against float NaN
    if answer_options is not None and not isinstance(answer_options, (str, list)):
        answer_options = None

    options_block = ""
    if answer_options:
        if isinstance(answer_options, str):
            answer_options = json.loads(answer_options)
        options_block = "\n".join(
            f"{chr(65+i)}. {opt}" for i, opt in enumerate(answer_options)
        )
        options_block = f"\n\n{options_block}"
    
    # Randomize peer order
    rng = random.Random(ordering_seed)
    shuffled = list(filtered_peers)
    rng.shuffle(shuffled)
    
    if shuffled:
        # Include peers with confidence labels
        peer_block = "\n\n".join(
            f"{p['agent_identifier']} (confidence {p.get('extracted_self_reported_confidence_integer', '?')}) "
            f"said: {p['raw_response_text']}"
            for p in shuffled
        )
    else:
        # Empty peer block
        peer_block = C5_EMPTY_PEERS_NOTE
    
    user = (
        f"Question: {question_text}{options_block}\n\n"
        f"{C5_INSTRUCTION_PREFIX}\n\n"
        f"{peer_block}\n\n"
        f"{C5_FINAL_INSTRUCTION}"
    )
    
    return system, user


def run_round1_confidence_weighted(
    focal_agent_info: Dict[str, Any],
    dumb_agents_info: List[Dict[str, Any]],
    round0_responses: List[Dict[str, Any]],
    question: Dict,
    judge_cascade: JudgeCascade,
    ordering_seed: int,
    confidence_threshold: int = 60,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Execute C5 Round 1 for the focal smart agent only.
    Dumb agents do NOT see filtered context — only the focal agent does.
    
    Returns:
        (responses, count_filtered_out)
    """
    focal_identifier = focal_agent_info["identifier"]
    
    # Filter peers by confidence
    filtered_peers, filtered_out_count = filter_peers_by_confidence(
        round0_responses, focal_identifier, confidence_threshold
    )
    
    # Build C5 prompt for focal agent
    system, user = build_c5_round1_prompt(
        question_text=question["question_text"],
        answer_options=question.get("answer_options"),
        filtered_peers=filtered_peers,
        ordering_seed=ordering_seed,
    )
    
    answer_options_str = ""
    if question.get("answer_options") and isinstance(question["answer_options"], (str, list)):
        opts = question["answer_options"]
        if isinstance(opts, str):
            opts = json.loads(opts)
        answer_options_str = ", ".join(f"{chr(65+i)}" for i in range(len(opts)))
    
    # Focal agent generates Round 1 response
    focal_agent = focal_agent_info["agent"]
    agent_response = focal_agent.generate_response(
        system_prompt=system,
        user_prompt=user,
        temperature=focal_agent_info.get("temperature", 0.7),
        maximum_output_tokens=focal_agent_info.get("max_tokens", 600),
        request_metadata={
            "question_id": question["question_identifier"],
            "round": 1,
            "condition": "C5",
        },
    )
    
    answer, method = extract_answer_with_fallback(
        agent_response.raw_text_output, question["question_text"],
        answer_options_str, judge_cascade,
    )
    confidence, conf_status = extract_confidence_regex(agent_response.raw_text_output)
    
    peer_context = json.dumps([
        {
            "agent": p["agent_identifier"],
            "text": p["raw_response_text"][:200],
            "confidence": p.get("extracted_self_reported_confidence_integer"),
        }
        for p in filtered_peers
    ])
    
    responses = [{
        "agent_identifier": focal_identifier,
        "role": "smart_focal",
        "model_name": agent_response.model_name_returned_by_provider,
        "provider": focal_agent_info.get("provider", "unknown"),
        "raw_response_text": agent_response.raw_text_output,
        "extracted_final_answer": answer,
        "extracted_self_reported_confidence_integer": confidence,
        "confidence_parse_status": conf_status,
        "answer_extraction_method": method,
        "peer_messages_seen_in_context": peer_context,
        "peer_messages_filtered_out_count": filtered_out_count,
        "peer_messages_ordering_seed": ordering_seed,
        "total_input_tokens": agent_response.total_input_tokens,
        "total_output_tokens": agent_response.total_output_tokens,
        "wall_clock_latency_seconds": agent_response.wall_clock_latency_seconds,
        "error_status": agent_response.error_status,
        "retry_attempts_used": agent_response.retry_attempts_used,
    }]
    
    # Dumb agents get standard Round 1 (they don't see filtered context)
    # Per §7.5: "The dumb agents in C5 do NOT see filtered peer messages"
    for dumb_info in dumb_agents_info:
        dumb_agent = dumb_info["agent"]
        
        # Build standard peer context for dumb agents
        dumb_peers = [
            {
                "agent_identifier": r["agent_identifier"],
                "response_text": r["raw_response_text"],
                "confidence": r.get("extracted_self_reported_confidence_integer"),
            }
            for r in round0_responses
            if r["agent_identifier"] != dumb_info["identifier"]
        ]
        
        from .debate_protocol import build_round1_prompt
        d_system, d_user = build_round1_prompt(
            question_text=question["question_text"],
            answer_options=question.get("answer_options"),
            peer_responses=dumb_peers,
            ordering_seed=ordering_seed,
        )
        
        d_response = dumb_agent.generate_response(
            system_prompt=d_system,
            user_prompt=d_user,
            temperature=dumb_info.get("temperature", 0.9),
            maximum_output_tokens=dumb_info.get("max_tokens", 350),
        )
        
        d_answer, d_method = extract_answer_with_fallback(
            d_response.raw_text_output, question["question_text"],
            answer_options_str, judge_cascade,
        )
        d_confidence, d_conf_status = extract_confidence_regex(d_response.raw_text_output)
        
        responses.append({
            "agent_identifier": dumb_info["identifier"],
            "role": "dumb",
            "model_name": d_response.model_name_returned_by_provider,
            "provider": dumb_info.get("provider", "local_huggingface"),
            "raw_response_text": d_response.raw_text_output,
            "extracted_final_answer": d_answer,
            "extracted_self_reported_confidence_integer": d_confidence,
            "confidence_parse_status": d_conf_status,
            "answer_extraction_method": d_method,
            "peer_messages_seen_in_context": None,
            "peer_messages_filtered_out_count": None,
            "peer_messages_ordering_seed": ordering_seed,
            "total_input_tokens": d_response.total_input_tokens,
            "total_output_tokens": d_response.total_output_tokens,
            "wall_clock_latency_seconds": d_response.wall_clock_latency_seconds,
            "error_status": d_response.error_status,
            "retry_attempts_used": d_response.retry_attempts_used,
        })
    
    return responses, filtered_out_count
