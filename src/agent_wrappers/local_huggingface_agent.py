"""
local_huggingface_agent.py — Local model agent for Llama 3.1 8B and Gemma 3/2.

Loads models at process start and keeps them in GPU memory.
Uses bfloat16 with flash_attention_2. No quantization.
Implements Gemma 3 fallback procedure per spec §2.6.
"""

import os
import time
import logging
import threading
from typing import Optional, Dict, Any

import torch

from .base_agent import BaseAgent, AgentResponse

logger = logging.getLogger("platos_ship.agents.local_hf")

# Global model cache — models loaded once, shared across all LocalHuggingFaceAgent instances
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _load_model_and_tokenizer(
    repo_id: str,
    fallback_repo_id: Optional[str],
    device: str,
    hf_token: Optional[str],
    attn_implementation: str = "flash_attention_2",
):
    """Load a model with fallback logic for Gemma 3 compatibility."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    actually_loaded_repo = repo_id
    fallback_was_triggered = False

    try:
        logger.info(f"Loading model: {repo_id} on {device} with {attn_implementation}")
        tokenizer = AutoTokenizer.from_pretrained(repo_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            device_map=device,
            token=hf_token,
        )
        logger.info(f"Successfully loaded {repo_id}")
    except (KeyError, ValueError, OSError) as e:
        error_msg = str(e).lower()
        if fallback_repo_id and ("unknown" in error_msg or "architecture" in error_msg
                                  or "config" in error_msg or "gemma3" in error_msg):
            logger.warning(
                f"Failed to load {repo_id}: {e}. Falling back to {fallback_repo_id}"
            )
            tokenizer = AutoTokenizer.from_pretrained(fallback_repo_id, token=hf_token)
            model = AutoModelForCausalLM.from_pretrained(
                fallback_repo_id,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
                device_map=device,
                token=hf_token,
            )
            actually_loaded_repo = fallback_repo_id
            fallback_was_triggered = True
            logger.info(f"Loaded fallback model: {fallback_repo_id}")
        else:
            raise

    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, actually_loaded_repo, fallback_was_triggered


class LocalHuggingFaceAgent(BaseAgent):
    """
    Local HuggingFace model agent. Loads model once and keeps in GPU memory.
    Both Llama and Gemma are loaded in bfloat16 with flash_attention_2.
    """

    def __init__(
        self,
        agent_name: str,
        huggingface_repository: str,
        fallback_huggingface_repository: Optional[str] = None,
        device: str = "cuda:0",
        hf_token: Optional[str] = None,
        attn_implementation: str = "flash_attention_2",
    ):
        super().__init__(agent_name=agent_name, provider="local_huggingface")
        self.huggingface_repository = huggingface_repository
        self.fallback_repository = fallback_huggingface_repository
        self.device = device
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN", "")
        self.attn_implementation = attn_implementation

        # Track which model was actually loaded
        self.actually_loaded_repository = None
        self.fallback_was_triggered = False
        self._model = None
        self._tokenizer = None

    def load_model(self):
        """Load the model into GPU memory. Call once at process start."""
        global _MODEL_CACHE
        cache_key = self.huggingface_repository

        with _MODEL_CACHE_LOCK:
            if cache_key in _MODEL_CACHE:
                cached = _MODEL_CACHE[cache_key]
                self._model = cached["model"]
                self._tokenizer = cached["tokenizer"]
                self.actually_loaded_repository = cached["actually_loaded_repo"]
                self.fallback_was_triggered = cached["fallback_triggered"]
                logger.info(f"Using cached model for {cache_key}")
                return

            # Load inside the lock to prevent concurrent double-loading
            model, tokenizer, actual_repo, fallback = _load_model_and_tokenizer(
                repo_id=self.huggingface_repository,
                fallback_repo_id=self.fallback_repository,
                device=self.device,
                hf_token=self.hf_token,
                attn_implementation=self.attn_implementation,
            )

            _MODEL_CACHE[cache_key] = {
                "model": model,
                "tokenizer": tokenizer,
                "actually_loaded_repo": actual_repo,
                "fallback_triggered": fallback,
            }

        self._model = model
        self._tokenizer = tokenizer
        self.actually_loaded_repository = actual_repo
        self.fallback_was_triggered = fallback

        # Log VRAM usage
        if torch.cuda.is_available():
            alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.info(f"GPU memory after loading {actual_repo}: {alloc_gb:.2f} GB")

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        maximum_output_tokens: int,
        request_metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        if self._model is None:
            self.load_model()

        start_time = time.time()
        try:
            # Build chat messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Use chat template if available
            if hasattr(self._tokenizer, "apply_chat_template"):
                input_text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                input_text = f"{system_prompt}\n\n{user_prompt}"

            inputs = self._tokenizer(
                input_text, return_tensors="pt", truncation=True, max_length=4096
            ).to(self._model.device)
            input_token_count = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=maximum_output_tokens,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            # Decode only the new tokens
            new_tokens = outputs[0][input_token_count:]
            raw_output = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            output_token_count = len(new_tokens)

            elapsed = time.time() - start_time
            result = AgentResponse(
                raw_text_output=raw_output,
                wall_clock_latency_seconds=elapsed,
                total_input_tokens=input_token_count,
                total_output_tokens=output_token_count,
                model_name_returned_by_provider=self.actually_loaded_repository or self.huggingface_repository,
                error_status="success",
                retry_attempts_used=0,
            )
            self._track_usage(result)
            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Local generation error ({self.agent_name}): {type(e).__name__}: {e}")
            result = AgentResponse(
                raw_text_output="",
                wall_clock_latency_seconds=elapsed,
                error_status="failure",
                model_name_returned_by_provider=self.actually_loaded_repository or self.huggingface_repository,
            )
            self._track_usage(result)
            return result
