"""
Run RAGAS evaluation on the full DocSight pipeline.
Metrics: faithfulness, answer_relevancy, context_precision, context_recall

Run: python -m src.evaluation.run_ragas --domain [finance|medical|both]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import argparse
import logging
from pathlib import Path

import pandas as pd
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_eval_set(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Eval set not found: {path}\nRun generate_qa_pairs.py first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_pipeline_on_eval(eval_set: list[dict]) -> list[dict]:
    """Run each eval question through the full DocSight graph and collect answers."""
    from src.agents.graph import run_query

    results = []
    for i, item in enumerate(eval_set):
        log.info("[%d/%d] %s", i + 1, len(eval_set), item["question"][:80])
        state = run_query(item["question"])
        results.append({
            "question":       item["question"],
            "answer":         state["final_answer"],
            "contexts":       [s["content"] for s in state.get("all_sources", [])],
            "ground_truth":   item["ground_truth"],
        })
    return results


def evaluate_domain(domain: str, sample: int | None = None) -> pd.DataFrame:
    path     = config.EVAL_FINANCE_PATH if domain == "finance" else config.EVAL_MEDICAL_PATH
    eval_set = _load_eval_set(path)

    if sample:
        import random
        eval_set = random.sample(eval_set, min(sample, len(eval_set)))

    log.info("Running pipeline on %d %s questions...", len(eval_set), domain)
    results = _run_pipeline_on_eval(eval_set)

    # Build RAGAS dataset
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import HuggingFaceEmbeddings

    ragas_ds = Dataset.from_list(results)

    llm = ChatGroq(
        model=config.RAGAS_LLM_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0,
    )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    log.info("Running RAGAS evaluation...")
    scores = evaluate(
        dataset=ragas_ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    df = scores.to_pandas()
    out_path = config.EVAL_DIR / f"{domain}_ragas_results.csv"
    df.to_csv(out_path, index=False)

    log.info("\n=== RAGAS Results (%s) ===", domain)
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric in df.columns:
            log.info("  %-25s %.4f", metric, df[metric].mean())

    log.info("Full results → %s", out_path)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["finance", "medical", "both"], default="both")
    parser.add_argument("--sample", type=int, default=None, help="Evaluate on N samples (faster)")
    args = parser.parse_args()

    all_dfs = {}
    if args.domain in ("finance", "both"):
        all_dfs["finance"] = evaluate_domain("finance", args.sample)
    if args.domain in ("medical", "both"):
        all_dfs["medical"] = evaluate_domain("medical", args.sample)

    if len(all_dfs) > 1:
        summary = {
            domain: {
                col: df[col].mean()
                for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
                if col in df.columns
            }
            for domain, df in all_dfs.items()
        }
        log.info("\n=== Combined Summary ===")
        for domain, metrics in summary.items():
            log.info("  %s: %s", domain, {k: f"{v:.3f}" for k, v in metrics.items()})
