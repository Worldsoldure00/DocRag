"""
Ablation study: BM25-only vs Dense-only vs Hybrid retrieval.
Measures RAGAS context_precision and context_recall for each retrieval strategy.

Run: python -m src.evaluation.ablation --domain [finance|medical]
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


def _get_retrievers(domain: str):
    from src.data_ingestion.embedder import load_finance_index, load_medical_index
    from src.data_ingestion.chunker import jsonl_to_documents
    from src.retrieval.hybrid_retriever import build_hybrid_retriever
    from langchain_community.retrievers import BM25Retriever

    if domain == "finance":
        index = load_finance_index()
        docs  = jsonl_to_documents(config.FINANCE_CHUNKS_PATH)
    else:
        index = load_medical_index()
        docs  = jsonl_to_documents(config.MEDICAL_CHUNKS_PATH)

    bm25_only   = BM25Retriever.from_documents(docs, k=config.TOP_K)
    dense_only  = index.as_retriever(search_kwargs={"k": config.TOP_K})
    hybrid      = build_hybrid_retriever(docs, index, k=config.TOP_K)

    return {"bm25": bm25_only, "dense": dense_only, "hybrid": hybrid}


def _retrieve_contexts(retriever, questions: list[str]) -> list[list[str]]:
    all_contexts = []
    for q in questions:
        docs = retriever.invoke(q)
        all_contexts.append([d.page_content for d in docs])
    return all_contexts


def run_ablation(domain: str, n: int = 50) -> pd.DataFrame:
    eval_path = config.EVAL_FINANCE_PATH if domain == "finance" else config.EVAL_MEDICAL_PATH
    with open(eval_path, encoding="utf-8") as f:
        eval_set = json.load(f)

    import random
    eval_set = random.sample(eval_set, min(n, len(eval_set)))
    questions     = [e["question"] for e in eval_set]
    ground_truths = [e["ground_truth"] for e in eval_set]

    retrievers = _get_retrievers(domain)

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import HuggingFaceEmbeddings

    llm = ChatGroq(model=config.RAGAS_LLM_MODEL, api_key=config.GROQ_API_KEY, temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    results_list = []
    for strategy, retriever in retrievers.items():
        log.info("Evaluating retrieval strategy: %s", strategy)
        contexts = _retrieve_contexts(retriever, questions)

        ds = Dataset.from_dict({
            "question":     questions,
            "contexts":     contexts,
            "ground_truth": ground_truths,
        })

        scores = evaluate(
            dataset=ds,
            metrics=[context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
        )
        df = scores.to_pandas()
        results_list.append({
            "strategy":         strategy,
            "context_precision": df["context_precision"].mean(),
            "context_recall":   df["context_recall"].mean(),
        })

    summary_df = pd.DataFrame(results_list)
    out_path = config.EVAL_DIR / f"{domain}_ablation.csv"
    summary_df.to_csv(out_path, index=False)

    log.info("\n=== Ablation Results (%s) ===", domain)
    log.info("\n%s", summary_df.to_string(index=False))
    log.info("Results → %s", out_path)
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["finance", "medical"], default="finance")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    run_ablation(args.domain, args.n)
