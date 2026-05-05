"""
Hybrid BM25 + dense FAISS retriever for a single domain.

Uses a self-contained RRF (Reciprocal Rank Fusion) implementation to avoid
importing from langchain_classic, whose __init__ chain pulls in
langchain_text_splitters which crashes on Python 3.13.
"""
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import config


def build_hybrid_retriever(
    docs: list[Document],
    faiss_index: FAISS,
    k: int = config.TOP_K,
    bm25_weight: float = config.BM25_WEIGHT,
    dense_weight: float = config.DENSE_WEIGHT,
):
    """
    Returns a callable that does weighted RRF over BM25 + dense FAISS results.
    Replaces langchain EnsembleRetriever to avoid Python-3.13 import crashes.
    """
    bm25  = BM25Retriever.from_documents(docs, k=k)
    dense = faiss_index.as_retriever(search_kwargs={"k": k})

    def invoke(query: str) -> list[Document]:
        bm25_docs  = bm25.invoke(query)
        dense_docs = dense.invoke(query)
        return _rrf([bm25_docs, dense_docs], [bm25_weight, dense_weight], k=k)

    # Attach an invoke attribute so it behaves like a LangChain retriever
    invoke.invoke = invoke  # type: ignore[attr-defined]
    return invoke


def _rrf(
    doc_lists: list[list[Document]],
    weights: list[float],
    k: int = 5,
    c: int = 60,
) -> list[Document]:
    """Weighted Reciprocal Rank Fusion across multiple ranked lists."""
    from collections import defaultdict
    scores: dict[str, float] = defaultdict(float)
    seen: dict[str, Document] = {}

    for docs, weight in zip(doc_lists, weights):
        for rank, doc in enumerate(docs, start=1):
            key = doc.page_content
            scores[key] += weight / (rank + c)
            seen[key] = doc

    ranked = sorted(seen.values(), key=lambda d: scores[d.page_content], reverse=True)
    return ranked[:k]


def deduplicate(docs: list[Document], threshold: float = 0.95) -> list[Document]:
    """Remove near-duplicate chunks based on text overlap."""
    seen: list[str] = []
    unique: list[Document] = []
    for doc in docs:
        text = doc.page_content.strip()
        if not any(_jaccard(text, s) > threshold for s in seen):
            seen.append(text)
            unique.append(doc)
    return unique


def _jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
