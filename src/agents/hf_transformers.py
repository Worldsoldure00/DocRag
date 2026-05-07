"""
Minimal HuggingFace Transformers backend for local inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import config


@dataclass
class LLMResponse:
    content: str


def generate_text(
    model_id: str,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float = 0.95,
) -> str:
    tokenizer, model = _load_model(model_id)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[-1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_p": top_p,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    output_ids = model.generate(**inputs, **gen_kwargs)
    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def invoke_text(
    model_id: str,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float = 0.95,
) -> LLMResponse:
    return LLMResponse(
        content=generate_text(
            model_id,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    )


def _resolve_dtype():
    import torch

    dtype = (config.HF_DTYPE or "auto").lower()
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32
    return "auto"


@lru_cache(maxsize=4)
def _load_model(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _resolve_dtype()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=config.HF_TOKEN,
        trust_remote_code=config.HF_TRUST_REMOTE_CODE,
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=config.HF_DEVICE_MAP,
        token=config.HF_TOKEN,
        trust_remote_code=config.HF_TRUST_REMOTE_CODE,
    )
    model.eval()
    return tokenizer, model
