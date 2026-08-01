"""
vllm_focal_agent.py — Local vLLM focal agent with answer-distribution logging.

Experiment E9 (mechanistic probe). Serves one open-weight focal model (default
Llama-3.1-8B-Instruct) locally in bf16 via vLLM, and exposes two things:

  generate(system, user)                   -> free-form debate response (as usual)
  answer_distribution(system, user, cands) -> P(answer = c) for each candidate c

The second method is the mechanistic instrument. It appends "Final answer:" to
the prompt and reads the model's next-token probability for each candidate
answer token (A..J, or digit tokens for GSM8K). Comparing this distribution in
Round 0 (independent) vs Round 1 (after wrong peers) quantifies how much
probability mass the focal moves toward the peer-asserted WRONG answer — a
representation-level sycophancy signal beneath the behavioural flip.

Both methods have `_batch` variants. vLLM's throughput comes almost entirely
from continuous batching, so the probe issues one call per *stage* over
thousands of prompts rather than one call per prompt; the single-prompt methods
are thin wrappers kept for readability and tests.

# Implements the GPU logprob probe from Review_Fix.md E9.
# Requires: vllm, torch (CUDA). Runs on a single 24 GB card for an 8B focal.
"""

import math
import logging
from typing import Dict, List, Optional, Sequence

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

    # ── prompt rendering ───────────────────────────────────────────────────
    def _chat_prefix(self, system: str, user: str, assistant_prefix: str = "") -> str:
        """Render the chat template, leaving an open assistant turn we can continue."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text + assistant_prefix

    # ── free-form generation ───────────────────────────────────────────────
    def generate_batch(
        self,
        prompts: Sequence[tuple],
        temperature: float = 0.7,
        max_tokens: int = 600,
        seeds: Optional[Sequence[int]] = None,
    ) -> List[str]:
        """
        Generate for many (system, user) pairs in one vLLM call.

        `seeds` must vary across replications of the same prompt — vLLM's `seed`
        is per-request, so reusing one seed would make every replication of a
        question return byte-identical text and collapse the replication
        dimension of the experiment.
        """
        from vllm import SamplingParams
        if not prompts:
            return []
        rendered = [self._chat_prefix(s, u) for (s, u) in prompts]
        if seeds is None:
            seeds = [self.seed + i for i in range(len(rendered))]
        params = [
            SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=int(sd))
            for sd in seeds
        ]
        out = self.llm.generate(rendered, params, use_tqdm=True)
        return [o.outputs[0].text for o in out]

    def generate(self, system: str, user: str, temperature: float = 0.7,
                 max_tokens: int = 600, seed: Optional[int] = None) -> str:
        seeds = None if seed is None else [seed]
        return self.generate_batch([(system, user)], temperature, max_tokens, seeds)[0]

    # ── answer distribution (the mechanistic instrument) ───────────────────
    def _dist_from_logprobs(self, output, candidates: List[str]) -> Dict[str, float]:
        logprobs_obj = output.outputs[0].logprobs
        if not logprobs_obj:
            return {c: float("nan") for c in candidates}
        first_pos = logprobs_obj[0]  # dict: token_id -> Logprob

        id_to_lp = {tid: lp.logprob for tid, lp in first_pos.items()}
        id_to_str = {tid: self.tokenizer.decode([tid]).strip().upper() for tid in id_to_lp}

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

        # Softmax-normalise over candidates that were found; missing -> 0.
        present = {c: lp for c, lp in raw.items() if lp is not None}
        if not present:
            return {c: float("nan") for c in candidates}
        mx = max(present.values())
        exps = {c: math.exp(lp - mx) for c, lp in present.items()}
        z = sum(exps.values())
        return {c: (exps[c] / z if c in exps else 0.0) for c in candidates}

    def answer_distribution_batch(
        self,
        prompts: Sequence[tuple],
        candidate_sets: Sequence[List[str]],
        assistant_prefix: str = "Final answer:",
    ) -> List[Dict[str, float]]:
        """P(next answer token = candidate) for many prompts in one vLLM call."""
        from vllm import SamplingParams
        if not prompts:
            return []
        rendered = [
            self._chat_prefix(s, u, assistant_prefix=f" {assistant_prefix} ")
            for (s, u) in prompts
        ]
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, seed=self.seed)
        out = self.llm.generate(rendered, sp, use_tqdm=True)
        return [self._dist_from_logprobs(o, list(c)) for o, c in zip(out, candidate_sets)]

    def answer_distribution(
        self, system: str, user: str, candidates: List[str],
        assistant_prefix: str = "Final answer:",
    ) -> Dict[str, float]:
        return self.answer_distribution_batch(
            [(system, user)], [candidates], assistant_prefix
        )[0]
