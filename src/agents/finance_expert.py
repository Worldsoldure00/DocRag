"""
Finance expert agent — retrieves relevant SEC filing chunks and generates a
grounded answer using the fine-tuned Llama-3.1-8B (via Ollama) or Groq fallback.
"""
import logging
from functools import lru_cache
from langchain.schema import Document
import config

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a financial analyst with expertise in SEC filings, earnings reports,
and corporate finance. Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so explicitly.
Always cite specific data points (numbers, dates, company names) from the context."""

PROMPT_TEMPLATE = """{system}

Context:
{context}

Question: {question}

Answer:"""


def _get_llm():
    if config.LLM_BACKEND == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=config.GROQ_EXPERT_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.1,
            max_tokens=1024,
        )
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=config.OLLAMA_FINANCE,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=1024,
    )


@lru_cache(maxsize=1)
def _get_retriever():
    """Lazy-load FAISS index + build hybrid retriever. Cached after first call."""
    from src.data_ingestion.embedder import load_finance_index
    from src.data_ingestion.chunker import jsonl_to_documents
    from src.retrieval.hybrid_retriever import build_hybrid_retriever

    index = load_finance_index()
    docs  = jsonl_to_documents(config.FINANCE_CHUNKS_PATH)
    return build_hybrid_retriever(docs, index, k=config.TOP_K)


def run(query: str, use_reranker: bool = True) -> dict:
    """
    Retrieve finance context and generate an answer.
    Returns dict with keys: answer, sources (list of Document dicts)
    """
    retriever = _get_retriever()
    raw_docs: list[Document] = retriever.invoke(query)

    if use_reranker:
        from src.retrieval.reranker import rerank
        raw_docs = rerank(query, raw_docs, top_k=config.TOP_K)

    context = "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('ticker','?')} | {d.metadata.get('source','?')}]\n{d.page_content}"
        for d in raw_docs
    )

    prompt = PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT, context=context, question=query
    )

    llm    = _get_llm()
    result = llm.invoke(prompt)
    answer = result.content if hasattr(result, "content") else str(result)

    log.info("Finance expert generated answer (%d chars)", len(answer))
    return {
        "answer":  answer,
        "sources": [{"content": d.page_content, "metadata": d.metadata} for d in raw_docs],
    }
