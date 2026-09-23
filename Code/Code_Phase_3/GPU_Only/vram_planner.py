#!/usr/bin/env python3
"""
vram_planner.py — what GPU the output-distribution probe (X8) actually needs.

Only X8 touches a GPU; every other experiment is API-bound and needs none.

The probe reads the model's probability distribution over answer options, so
two constraints are unusual compared with ordinary inference:

  1. **Precision matters.** A 4-bit quantised model does not have the same
     output distribution as the bf16 model. For a probe whose entire purpose
     is measuring probability mass, quantisation is a confound, not an
     optimisation. Plan in bf16.
  2. **Throughput barely matters.** The probe is one batch per stage over
     2,700 trials, not a live service. A card that merely fits the model
     finishes in well under an hour.

    python3 GPU_Only/vram_planner.py
    python3 GPU_Only/vram_planner.py --model llama-3.1-70b --max-len 4096
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ModelSpec:
    name: str
    params_billions: float
    n_layers: int
    n_kv_heads: int          # GQA: key/value heads, not attention heads
    head_dim: int
    note: str = ""

    def weights_gb(self, bytes_per_param: float = 2.0) -> float:
        return self.params_billions * 1e9 * bytes_per_param / 1024**3

    def kv_bytes_per_token(self, bytes_per_element: float = 2.0) -> float:
        # K and V, per layer, per KV head, per head dimension.
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * bytes_per_element

    def kv_gb_per_sequence(self, max_len: int) -> float:
        return self.kv_bytes_per_token() * max_len / 1024**3


MODELS: Dict[str, ModelSpec] = {
    "llama-3.1-8b": ModelSpec(
        "Llama-3.1-8B-Instruct", 8.03, 32, 8, 128,
        "the probe model used in the released study"),
    "qwen2.5-7b": ModelSpec("Qwen2.5-7B-Instruct", 7.62, 28, 4, 128),
    "gemma-3-27b": ModelSpec("Gemma-3-27B-IT", 27.4, 62, 16, 128),
    "llama-3.1-70b": ModelSpec(
        "Llama-3.1-70B-Instruct", 70.6, 80, 8, 128,
        "a strong open-weight focal — addresses the paper's own limitation"),
    "qwen2.5-72b": ModelSpec("Qwen2.5-72B-Instruct", 72.7, 80, 8, 128),
}

# Cards worth considering, with usable VRAM.
CARDS: List[tuple] = [
    ("RTX 4090 / L4", 24),
    ("RTX 5090", 32),
    ("A100-40GB", 40),
    ("L40S / A6000 / RTX 6000 Ada", 48),
    ("A100-80GB / H100-80GB", 80),
    ("2x A100-80 / 2x H100", 160),
]

PRECISIONS = {"bf16": 2.0, "fp8": 1.0, "int4 (awq/gptq)": 0.5}


def plan(model: ModelSpec, max_len: int, utilisation: float = 0.90) -> None:
    print(f"\n{'=' * 74}")
    print(f"{model.name}   {model.params_billions}B params")
    if model.note:
        print(f"  ({model.note})")
    print("=" * 74)

    kv_per_token_kb = model.kv_bytes_per_token() / 1024
    kv_per_seq_gb = model.kv_gb_per_sequence(max_len)
    print(f"KV cache: {kv_per_token_kb:.0f} KB per token"
          f"  ->  {kv_per_seq_gb * 1024:.0f} MB per {max_len}-token sequence")

    print(f"\n{'precision':<18}{'weights':>10}{'min card':>12}"
          f"{'comfortable':>14}   distribution fidelity")
    print("-" * 74)
    for label, bytes_per_param in PRECISIONS.items():
        weights = model.weights_gb(bytes_per_param)
        # Minimum: weights plus room for a couple of concurrent sequences.
        minimum = (weights + 2 * kv_per_seq_gb) / utilisation
        # Comfortable: weights plus ~32 concurrent sequences.
        comfortable = (weights + 32 * kv_per_seq_gb) / utilisation
        fidelity = ("exact" if label == "bf16"
                    else "ALTERED — do not use for the probe")
        print(f"{label:<18}{weights:>9.1f}G{minimum:>11.0f}G"
              f"{comfortable:>13.0f}G   {fidelity}")

    weights_bf16 = model.weights_gb(2.0)
    print(f"\n{'card':<32}{'VRAM':>7}{'fits bf16?':>12}{'concurrent seqs':>18}")
    print("-" * 74)
    for card_name, vram in CARDS:
        usable = vram * utilisation
        headroom = usable - weights_bf16
        if headroom <= kv_per_seq_gb:
            verdict, seqs = "no", "-"
        else:
            n = int(headroom / kv_per_seq_gb)
            verdict = "yes"
            seqs = f"~{n}"
        print(f"{card_name:<32}{vram:>6}G{verdict:>12}{seqs:>18}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama-3.1-8b",
                        choices=sorted(MODELS) + ["all"])
    parser.add_argument("--max-len", type=int, default=4096,
                        help="vLLM max_model_len (the probe uses 4096)")
    parser.add_argument("--utilisation", type=float, default=0.90,
                        help="vLLM gpu_memory_utilization")
    args = parser.parse_args()

    chosen = sorted(MODELS) if args.model == "all" else [args.model]
    for key in chosen:
        plan(MODELS[key], args.max_len, args.utilisation)

    print(f"\n{'=' * 74}")
    print("Notes")
    print("=" * 74)
    print("* Only X8 needs a GPU. X1-X7 are API-bound and need none.")
    print("* The probe runs one batch per stage over 2,700 trials, so a card")
    print("  that fits the model finishes in well under an hour. Buying VRAM")
    print("  for throughput is wasted here; buying it to fit bf16 is not.")
    print("* Quantisation changes the output distribution. The probe measures")
    print("  probability mass, so int4/fp8 would confound the measurement.")
    print("  Report the precision used, whatever it is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
