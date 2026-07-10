"""
vllm_focal_agent.py — Local vLLM focal agent with answer-distribution logging.

Experiment E9 (mechanistic probe). Serves one open-weight focal model (default
Llama-3.1-8B-Instruct) locally in bf16 via vLLM, and exposes two things:

  generate(system, user)                 -> free-form debate response (as usual)
  answer_distribution(system, user, cands) -> P(answer = c) for each candidate c

The second method is the mechanistic instrument. It appends "Final answer:" to
the prompt and reads the model's next-token probability for each candidate
answer token (A..J, or digit tokens for GSM8K). Comparing this distribution in
Round 0 (independent) vs Round 1 (after wrong peers) quantifies how much
probability mass the focal moves toward the peer-asserted WRONG answer — a
representation-level sycophancy signal beneath the behavioural flip.

# Implements the GPU logprob probe from Review_Fix.md E9.
# Requires: vllm, torch (CUDA). Runs on a single 24 GB card (RTX 4090) for 8B.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("platos_ship.vllm_focal")


class VLLMFocalAgent:
    def __init__(
        self,
        model_repo: str = "meta-llama/Llama-3.1-8B-Instruct",
        dtype: str = "bfloat16",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
        seed: int = 20260502,
        hf_token: Optional[str] = None,
    ):
        from vllm import LLM  # imported lazily so CPU machines can import this module
        self.model_repo = model_repo
        self.provider = "vllm_local"
        self.model_name = model_repo
        self.seed = seed
        logger.info(f"Loading {model_repo} in {dtype} via vLLM (max_len={max_model_len})")
        self.llm = LLM(
            model=model_repo,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def _chat_prefix(self, system: str, user: str, assistant_prefix: str = "") -> str:
        """Render the chat template, leaving an open assistant turn we can continue."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text + assistant_prefix

    def generate(self, system: str, user: str, temperature: float = 0.7,
                 max_tokens: int = 600) -> str:
        from vllm import SamplingParams
        prompt = self._chat_prefix(system, user)
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=self.seed)
        out = self.llm.generate([prompt], sp, use_tqdm=False)
        return out[0].outputs[0].text

    def answer_distribution(
        self, system: str, user: str, candidates: List[str],
        assistant_prefix: str = "Final answer:",
    ) -> Dict[str, float]:
        """
        P(next answer token = candidate) for each candidate string.

        Reads the next-token logprob distribution right after "Final answer:".
        Candidates are matched on their FIRST token id (a single letter A..J, or
        the leading digit of a numeric answer), then softmax-normalised over the
        candidate set so the returned values sum to 1 across `candidates`.
        """
        from vllm import SamplingParams
        prompt = self._chat_prefix(system, user, assistant_prefix=f" {assistant_prefix} ")
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, seed=self.seed)
        out = self.llm.generate([prompt], sp, use_tqdm=False)

        logprobs_obj = out[0].outputs[0].logprobs
        if not logprobs_obj:
            return {c: float("nan") for c in candidates}
        first_pos = logprobs_obj[0]  # dict: token_id -> Logprob

        # Map token_id -> logprob and token_id -> decoded string
        id_to_lp = {tid: lp.logprob for tid, lp in first_pos.items()}
        id_to_str = {}
        for tid in id_to_lp:
            id_to_str[tid] = self.tokenizer.decode([tid]).strip().upper()

        raw = {}
        for cand in candidates:
            key = str(cand).strip().upper()[:1]  # first char (letter or leading digit)
            best_lp = None
            for tid, s in id_to_str.items():
                if s[:1] == key:
                    lp = id_to_lp[tid]
                    if best_lp is None or lp > best_lp:
                        best_lp = lp
            raw[cand] = best_lp

        # Softmax-normalise over candidates that were found; missing -> very low
        present = {c: lp for c, lp in raw.items() if lp is not None}
        if not present:
            return {c: float("nan") for c in candidates}
        mx = max(present.values())
        exps = {c: math.exp(lp - mx) for c, lp in present.items()}
        z = sum(exps.values())
        dist = {c: (exps[c] / z if c in exps else 0.0) for c in candidates}
        return dist
