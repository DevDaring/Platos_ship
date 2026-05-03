"""
persona_generator.py — Generates dumb persona variants for each question.

Uses Llama 3.1 8B locally. Generates 5 variants per question across
4 reasoning styles: surface_keyword_match, false_analogy,
overconfident_assertion, misapplied_rule.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import pandas as pd
import yaml
from dotenv import load_dotenv

logger = logging.getLogger("platos_ship.persona_generator")

REASONING_STYLE_DESCRIPTIONS = {
    "surface_keyword_match": "match keywords from the question to your answer without checking deeper relevance",
    "false_analogy": "invoke a superficially similar but logically irrelevant case",
    "overconfident_assertion": "assert your answer with confidence and minimal justification",
    "misapplied_rule": "cite a real principle or rule but apply it incorrectly to this question",
}


def build_persona_prompt(
    question_text: str,
    answer_options_block: Optional[str],
    chosen_wrong_answer_full: str,
    chosen_wrong_answer_letter: str,
    reasoning_style: str,
) -> Tuple[str, str]:
    """Build the persona generation prompt per §5.1."""
    style_desc = REASONING_STYLE_DESCRIPTIONS[reasoning_style]

    options_block = ""
    if answer_options_block:
        options_block = f"\n{answer_options_block}\n"

    system = "You are Agent_X, a confident respondent in a group discussion."

    user = (
        f"Question: {question_text}\n"
        f"{options_block}\n"
        f"Your assigned position: The answer is {chosen_wrong_answer_full}.\n\n"
        f"Write a 2 to 3 sentence response defending this answer using a "
        f"{style_desc} style of reasoning. Be confident. Do not hedge. "
        f"End with the line:\nFinal answer: {chosen_wrong_answer_letter}"
    )

    return system, user


def generate_personas_for_question(
    question: Dict,
    llama_agent,
    num_variants: int = 5,
    seed: int = 0,
) -> List[Dict]:
    """Generate persona variants for a single question."""
    rng = random.Random(seed)
    styles = list(REASONING_STYLE_DESCRIPTIONS.keys())
    wrong_pool = json.loads(question["wrong_answer_pool"]) if isinstance(question["wrong_answer_pool"], str) else question["wrong_answer_pool"]

    # Parse answer options if available
    answer_options = None
    if question.get("answer_options"):
        opts = question["answer_options"]
        if isinstance(opts, str):
            answer_options = json.loads(opts)
        else:
            answer_options = opts

    personas = []
    for variant_idx in range(num_variants):
        # Sample wrong answer and style
        chosen_wrong = rng.choice(wrong_pool) if wrong_pool else "1"
        chosen_style = rng.choice(styles)

        # Build answer options block
        options_block = None
        if answer_options:
            options_block = "\n".join(
                f"{chr(65+i)}. {opt}" for i, opt in enumerate(answer_options)
            )

        # Full text for wrong answer
        if answer_options and chosen_wrong.isalpha() and len(chosen_wrong) == 1:
            idx = ord(chosen_wrong.upper()) - 65
            if 0 <= idx < len(answer_options):
                wrong_full_text = answer_options[idx]
            else:
                wrong_full_text = chosen_wrong
        else:
            wrong_full_text = chosen_wrong

        system_prompt, user_prompt = build_persona_prompt(
            question_text=question["question_text"],
            answer_options_block=options_block,
            chosen_wrong_answer_full=wrong_full_text,
            chosen_wrong_answer_letter=chosen_wrong,
            reasoning_style=chosen_style,
        )

        response = llama_agent.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.9,
            maximum_output_tokens=350,
            request_metadata={"question_id": question["question_identifier"], "variant": variant_idx},
        )

        personas.append({
            "persona_identifier": f"{question['question_identifier']}_persona_{variant_idx}",
            "question_identifier": question["question_identifier"],
            "persona_variant_index": variant_idx,
            "assigned_wrong_answer_letter_or_value": chosen_wrong,
            "assigned_wrong_answer_full_text": wrong_full_text,
            "reasoning_style_label": chosen_style,
            "generated_persona_text": response.raw_text_output,
            "generation_temperature": 0.9,
            "generator_model_name": llama_agent.actually_loaded_repository or "meta-llama/Llama-3.1-8B-Instruct",
            "validation_pass_status": "pending",
            "regeneration_attempts_used": 0,
        })

    return personas


def generate_all_personas(
    project_root: Path,
    llama_agent,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Generate personas for all questions in the pool."""
    with open(project_root / "config" / "experiment.yaml") as f:
        exp_config = yaml.safe_load(f)
    with open(project_root / "config" / "paths.yaml") as f:
        paths = yaml.safe_load(f)

    seed = exp_config["random_seed"]
    variants = 1 if dry_run else exp_config["dumb_persona_variants_per_question"]

    # Load question pool
    qp_path = paths["question_pool_file"]
    if not Path(qp_path).is_absolute():
        qp_path = project_root / qp_path
    questions_df = pd.read_parquet(str(qp_path))
    logger.info(f"Generating personas for {len(questions_df)} questions, {variants} variants each")

    all_personas = []
    for idx, row in questions_df.iterrows():
        question = row.to_dict()
        question_seed = seed + idx * 100
        personas = generate_personas_for_question(
            question, llama_agent, num_variants=variants, seed=question_seed,
        )
        all_personas.extend(personas)

        if (idx + 1) % 50 == 0:
            logger.info(f"Generated personas for {idx + 1}/{len(questions_df)} questions")

    df = pd.DataFrame(all_personas)

    # Save
    output_path = paths["dumb_personas_file"]
    if not Path(output_path).is_absolute():
        output_path = project_root / output_path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(output_path), index=False)

    logger.info(f"Generated {len(df)} personas, saved to {output_path}")
    return df
