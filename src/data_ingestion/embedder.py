"""
Builds and saves FAISS indexes for finance and medical domains.
Run: python -m src.data_ingestion.embedder [--domain finance|medical|both]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import argparse
import logging

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import config
from src.data_ingestion.chunker import jsonl_to_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_embedder(model_name: str) -> HuggingFaceEmbeddings:
    log.info("Loading embedding model: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda" if _cuda_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _subsample(docs, max_n):
    """Stratified subsample by ticker/source to keep representation."""
    if max_n is None or len(docs) <= max_n:
        return docs
    import random
    # Group by ticker/source, sample proportionally
    from collections import defaultdict
    groups = defaultdict(list)
    for d in docs:
        key = d.metadata.get("ticker") or d.metadata.get("source", "unknown")
        groups[key].append(d)
    sampled = []
    per_group = max(1, max_n // len(groups))
    for key, group_docs in groups.items():
        sampled.extend(random.sample(group_docs, min(per_group, len(group_docs))))
    # Top up to max_n if under
    sampled_idx = {id(d) for d in sampled}
    remaining = [d for d in docs if id(d) not in sampled_idx]
    if len(sampled) < max_n and remaining:
        sampled.extend(random.sample(remaining, min(max_n - len(sampled), len(remaining))))
    log.info("  Subsampled %d → %d chunks", len(docs), len(sampled))
    return sampled[:max_n]


def build_finance_index() -> FAISS:
    log.info("Building finance FAISS index...")
    docs = jsonl_to_documents(config.FINANCE_CHUNKS_PATH)
    log.info("  %d finance chunks loaded", len(docs))
    docs = _subsample(docs, getattr(config, "MAX_FINANCE_CHUNKS", None))

    embedder = _load_embedder(config.FINANCE_EMBED_MODEL)
    index = FAISS.from_documents(docs, embedder)

    config.FINANCE_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    index.save_local(str(config.FINANCE_INDEX_PATH))
    log.info("Finance index saved → %s  (%d vectors)", config.FINANCE_INDEX_PATH, len(docs))
    return index


def build_medical_index() -> FAISS:
    log.info("Building medical FAISS index...")
    docs = jsonl_to_documents(config.MEDICAL_CHUNKS_PATH)
    log.info("  %d medical chunks loaded", len(docs))
    docs = _subsample(docs, getattr(config, "MAX_MEDICAL_CHUNKS", None))

    embedder = _load_embedder(config.MEDICAL_EMBED_MODEL)
    index = FAISS.from_documents(docs, embedder)

    config.MEDICAL_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    index.save_local(str(config.MEDICAL_INDEX_PATH))
    log.info("Medical index saved → %s  (%d vectors)", config.MEDICAL_INDEX_PATH, len(docs))
    return index


def load_finance_index() -> FAISS:
    embedder = _load_embedder(config.FINANCE_EMBED_MODEL)
    return FAISS.load_local(
        str(config.FINANCE_INDEX_PATH),
        embedder,
        allow_dangerous_deserialization=True,
    )


def load_medical_index() -> FAISS:
    embedder = _load_embedder(config.MEDICAL_EMBED_MODEL)
    return FAISS.load_local(
        str(config.MEDICAL_INDEX_PATH),
        embedder,
        allow_dangerous_deserialization=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["finance", "medical", "both"], default="both")
    args = parser.parse_args()

    if args.domain in ("finance", "both"):
        build_finance_index()
    if args.domain in ("medical", "both"):
        build_medical_index()
