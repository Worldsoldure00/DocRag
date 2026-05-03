"""
Hybrid BM25 + dense FAISS retriever for a single domain.
"""
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import config


def build_hybrid_retriever(
    docs: list[Document],
    faiss_index: FAISS,
    k: int = config.TOP_K,
    bm25_weight: float = config.BM25_WEIGHT,
    dense_weight: float = config.DENSE_WEIGHT,
) -> EnsembleRetriever:
    """
    Combines BM25 keyword search with dense FAISS vector search.
    bm25_weight + dense_weight should sum to 1.0.
    """
    bm25   = BM25Retriever.from_documents(docs, k=k)
    dense  = faiss_index.as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[bm25_weight, dense_weight],
    )


def deduplicate(docs: list[Document], threshold: float = 0.95) -> list[Document]:
    """Remove near-duplicate chunks based on text overlap."""
    seen: list[str] = []
    unique: list[Document] = []
    for doc in docs:
        text = doc.page_content.strip()
        is_dup = any(
            _jaccard(text, s) > threshold for s in seen
        )
        if not is_dup:
            seen.append(text)
            unique.append(doc)
    return unique


def _jaccard(a: str, b: str) -> float:
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
