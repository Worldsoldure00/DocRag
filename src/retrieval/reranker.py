"""
Optional cross-encoder reranker to re-score retrieved chunks.
Uses a lightweight cross-encoder that runs on CPU in ~100ms per query.
"""
from langchain_core.documents import Document


_reranker = None  # lazy-loaded


def _get_reranker():
    global _reranker
    if _reranker is None:
        import os
        from sentence_transformers import CrossEncoder
        force_cpu = os.environ.get("DOCSIGHT_FORCE_CPU") == "1"
        device = "cpu" if force_cpu else None  # None = auto-detect
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
    return _reranker


def rerank(query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
    """Re-rank docs by cross-encoder relevance score and return top_k."""
    if not docs:
        return docs

    reranker = _get_reranker()
    pairs    = [(query, doc.page_content) for doc in docs]
    scores   = reranker.predict(pairs)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    # Drop irrelevant documents (ms-marco cross-encoder outputs logits; negative means irrelevant)
    relevant_docs = [doc for score, doc in ranked[:top_k] if score > 0.0]
    return relevant_docs
