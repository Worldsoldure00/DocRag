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
from langchain_community.embeddings import HuggingFaceEmbeddings

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


def build_finance_index() -> FAISS:
    log.info("Building finance FAISS index...")
    docs = jsonl_to_documents(config.FINANCE_CHUNKS_PATH)
    log.info("  %d finance chunks loaded", len(docs))

    embedder = _load_embedder(config.FINANCE_EMBED_MODEL)
    index = FAISS.from_documents(docs, embedder)

    config.FINANCE_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    index.save_local(str(config.FINANCE_INDEX_PATH))
    log.info("Finance index saved → %s", config.FINANCE_INDEX_PATH)
    return index


def build_medical_index() -> FAISS:
    log.info("Building medical FAISS index...")
    docs = jsonl_to_documents(config.MEDICAL_CHUNKS_PATH)
    log.info("  %d medical chunks loaded", len(docs))

    embedder = _load_embedder(config.MEDICAL_EMBED_MODEL)
    index = FAISS.from_documents(docs, embedder)

    config.MEDICAL_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    index.save_local(str(config.MEDICAL_INDEX_PATH))
    log.info("Medical index saved → %s", config.MEDICAL_INDEX_PATH)
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
