"""
scopes.py — which dataset a question id belongs to.

One definition, used by the runner and the analysis. Pooling across scopes is
a defect: X4 adds GSM-Symbolic Round-0 answers for some models only, so a
per-model mean over the whole Round-0 cache gives those models a different
denominator from the rest. The capability gradient's x-axis was computed that
way before this module existed.
"""

from __future__ import annotations

MAIN_SCOPE = "main300"


def dataset_scope(question_id) -> str:
    """Dataset scope from the question id prefix; never pool across these."""
    qid = str(question_id)
    if qid.startswith("gsm8k_perturbed"):
        return "perturbed_quarantined"
    if qid.startswith("gsmsym_"):
        return "gsm_symbolic"
    if qid.startswith("gsmorig_"):
        return "gsm8k_matched_original"
    return MAIN_SCOPE


def main_scope_only(frame, column: str = "question_identifier"):
    """Rows whose question belongs to the main 300-item pool."""
    if frame.empty or column not in frame.columns:
        return frame
    return frame[frame[column].map(dataset_scope) == MAIN_SCOPE]
