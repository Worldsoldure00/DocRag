"""
Formats raw chunks into Alpaca/SFT training JSON for router, finance, and medical models.
Also generates synthetic QA pairs using Groq (llama-3.3-70b-versatile).

Run: python training/data_prep.py --task [router|finance|medical|all]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import argparse
import logging
import random
import time
from pathlib import Path

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TRAINING_DIR = config.BASE_DIR / "training" / "data"
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

# ── Synthetic QA generation ───────────────────────────────────────────────────

FINANCE_QA_PROMPT = """You are generating training data for a financial RAG model.
Given the following excerpt from an SEC 10-K filing, generate 3 question-answer pairs.
Each QA pair must be answerable ONLY from this context.

Context:
{context}

Output a JSON array of 3 objects: [{{"question": "...", "answer": "..."}}]
Output ONLY the JSON array, nothing else."""

MEDICAL_QA_PROMPT = """You are generating training data for a medical RAG model.
Given the following PubMed abstract, generate 3 question-answer pairs.
Each QA pair must be answerable ONLY from this context.

Context:
{context}

Output a JSON array of 3 objects: [{{"question": "...", "answer": "..."}}]
Output ONLY the JSON array, nothing else."""

ROUTER_EXAMPLES = [
    {"input": "What was Apple's revenue in Q3 2023?", "output": "finance"},
    {"input": "What are the side effects of metformin?", "output": "medical"},
    {"input": "How does JPMorgan's net income compare to Goldman Sachs?", "output": "finance"},
    {"input": "What is the mechanism of action of statins?", "output": "medical"},
    {"input": "What is Tesla's debt-to-equity ratio?", "output": "finance"},
    {"input": "What are the risk factors for type 2 diabetes?", "output": "medical"},
    {"input": "How did Pfizer's COVID-19 vaccine revenue affect their 2022 earnings?", "output": "both"},
    {"input": "What are the cardiovascular benefits of GLP-1 drugs and their market outlook?", "output": "both"},
    {"input": "Compare the R&D spending of JNJ and PFE in their 10-K filings", "output": "finance"},
    {"input": "What clinical trials are ongoing for Alzheimer's disease treatment?", "output": "medical"},
]


def _call_groq(prompt: str, retries: int = 3) -> str:
    from groq import Groq
    client = Groq(api_key=config.GROQ_API_KEY)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=config.GROQ_QA_GEN_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=800,
            )
            return resp.choices[0].message.content
        except Exception as e:
            log.warning("Groq call failed (attempt %d): %s", attempt + 1, e)
            time.sleep(2 ** attempt)
    return "[]"


def generate_qa_pairs(
    chunks_path: Path,
    prompt_template: str,
    n_chunks: int = 500,
    output_path: Path | None = None,
) -> list[dict]:
    """Generate QA pairs from chunk JSONL using Groq."""
    import json as _json

    with open(chunks_path, encoding="utf-8") as f:
        all_chunks = [_json.loads(line) for line in f if line.strip()]

    # Sample n_chunks (prefer text chunks over tables)
    text_chunks = [c for c in all_chunks if c["metadata"].get("type") == "text"]
    sample = random.sample(text_chunks, min(n_chunks, len(text_chunks)))

    pairs = []
    for i, chunk in enumerate(sample):
        context = chunk["text"][:1200]  # stay within token budget
        prompt  = prompt_template.format(context=context)
        raw     = _call_groq(prompt)

        try:
            qa_list = _json.loads(raw)
            for qa in qa_list:
                pairs.append({
                    "instruction": "Answer the question based on the provided context.",
                    "input":       f"Context:\n{context}\n\nQuestion: {qa['question']}",
                    "output":      qa["answer"],
                    "metadata":    chunk["metadata"],
                })
        except Exception as e:
            log.debug("Failed to parse QA for chunk %d: %s", i, e)

        if (i + 1) % 50 == 0:
            log.info("  generated %d QA pairs so far", len(pairs))
        time.sleep(0.1)  # gentle rate limit

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            _json.dump(pairs, f, indent=2)
        log.info("Saved %d QA pairs → %s", len(pairs), output_path)

    return pairs


# ── Format for SFT trainer ────────────────────────────────────────────────────

def format_for_sft(pairs: list[dict], output_path: Path) -> None:
    """Convert to Alpaca format expected by Unsloth SFTTrainer."""
    formatted = []
    for p in pairs:
        text = (
            f"<s>[INST] {p['instruction']}\n\n{p['input']} [/INST] {p['output']}</s>"
        )
        formatted.append({"text": text})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2)
    log.info("Formatted %d SFT examples → %s", len(formatted), output_path)


def prepare_router_data() -> None:
    """Build synthetic + manual router classification training set."""
    examples = list(ROUTER_EXAMPLES)

    # Extend with more synthetic examples via Groq
    extra_prompt = """Generate 50 diverse question-domain classification examples for a RAG router.
Domains: finance (SEC filings, earnings, stocks), medical (drugs, diseases, clinical), both (spans both).
Output JSON array: [{"input": "question here", "output": "finance|medical|both"}]
Make 60% finance, 30% medical, 10% both. Output ONLY the JSON array."""

    raw = _call_groq(extra_prompt)
    try:
        extra = json.loads(raw)
        examples.extend(extra)
    except Exception:
        log.warning("Could not parse extra router examples")

    formatted = []
    for ex in examples:
        text = (
            f"<s>[INST] Classify this query's domain.\n\n{ex['input']} [/INST] {ex['output']}</s>"
        )
        formatted.append({"text": text})

    out = TRAINING_DIR / "router_train.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2)
    log.info("Router training data: %d examples → %s", len(formatted), out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["router", "finance", "medical", "all"], default="all")
    parser.add_argument("--n-chunks", type=int, default=500, help="Chunks to sample per domain")
    args = parser.parse_args()

    if args.task in ("router", "all"):
        log.info("=== Preparing router training data ===")
        prepare_router_data()

    if args.task in ("finance", "all"):
        log.info("=== Generating finance QA pairs ===")
        pairs = generate_qa_pairs(
            config.FINANCE_CHUNKS_PATH,
            FINANCE_QA_PROMPT,
            n_chunks=args.n_chunks,
            output_path=TRAINING_DIR / "finance_qa.json",
        )
        format_for_sft(pairs, TRAINING_DIR / "finance_sft.json")

    if args.task in ("medical", "all"):
        log.info("=== Generating medical QA pairs ===")
        pairs = generate_qa_pairs(
            config.MEDICAL_CHUNKS_PATH,
            MEDICAL_QA_PROMPT,
            n_chunks=args.n_chunks,
            output_path=TRAINING_DIR / "medical_qa.json",
        )
        format_for_sft(pairs, TRAINING_DIR / "medical_sft.json")
