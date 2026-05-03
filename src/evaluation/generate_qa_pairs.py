"""
Generates evaluation QA pairs (ground-truth dataset for RAGAS).
Uses Groq llama-3.3-70b to create question + ground_truth + relevant_context.

Run: python -m src.evaluation.generate_qa_pairs --domain [finance|medical|both] --n 100
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import random
import argparse
import logging
import time
from pathlib import Path

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FINANCE_EVAL_PROMPT = """You are creating an evaluation dataset for a financial Q&A system.
Given this SEC filing excerpt, create ONE high-quality QA pair with a verifiable answer.

Context:
{context}

Output a single JSON object:
{{"question": "specific, factual question", "ground_truth": "precise answer with numbers/dates from context", "relevant": true}}
Output ONLY the JSON object."""

MEDICAL_EVAL_PROMPT = """You are creating an evaluation dataset for a medical Q&A system.
Given this biomedical text excerpt, create ONE high-quality QA pair.

Context:
{context}

Output a single JSON object:
{{"question": "specific clinical or scientific question", "ground_truth": "precise answer grounded in the context", "relevant": true}}
Output ONLY the JSON object."""


def _call_groq(prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=config.GROQ_API_KEY)
    try:
        resp = client.chat.completions.create(
            model=config.GROQ_QA_GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=400,
        )
        return resp.choices[0].message.content
    except Exception as e:
        log.warning("Groq error: %s", e)
        return "{}"


def generate_eval_set(domain: str, n: int = 100) -> list[dict]:
    """Generate n evaluation examples for given domain."""
    chunks_path = config.FINANCE_CHUNKS_PATH if domain == "finance" else config.MEDICAL_CHUNKS_PATH
    prompt_tpl  = FINANCE_EVAL_PROMPT if domain == "finance" else MEDICAL_EVAL_PROMPT

    with open(chunks_path, encoding="utf-8") as f:
        all_chunks = [json.loads(line) for line in f if line.strip()]

    text_chunks = [c for c in all_chunks if c["metadata"].get("type") in ("text", "abstract")]
    sample = random.sample(text_chunks, min(n * 2, len(text_chunks)))  # oversample, filter bad ones

    eval_set = []
    for chunk in sample:
        if len(eval_set) >= n:
            break

        context = chunk["text"][:1000]
        raw     = _call_groq(prompt_tpl.format(context=context))

        try:
            qa = json.loads(raw)
            if qa.get("question") and qa.get("ground_truth"):
                eval_set.append({
                    "question":     qa["question"],
                    "ground_truth": qa["ground_truth"],
                    "contexts":     [context],
                    "metadata":     chunk["metadata"],
                })
        except Exception:
            pass

        time.sleep(0.15)

    log.info("Generated %d eval pairs for domain=%s", len(eval_set), domain)
    return eval_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["finance", "medical", "both"], default="both")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)

    if args.domain in ("finance", "both"):
        data = generate_eval_set("finance", args.n)
        with open(config.EVAL_FINANCE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        log.info("Finance eval → %s", config.EVAL_FINANCE_PATH)

    if args.domain in ("medical", "both"):
        data = generate_eval_set("medical", args.n)
        with open(config.EVAL_MEDICAL_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        log.info("Medical eval → %s", config.EVAL_MEDICAL_PATH)
