"""
Medical expert agent — retrieves PubMed/PubMedQA chunks and generates a
grounded answer using fine-tuned BioMistral-7B (via Ollama) or Groq fallback.
"""
import logging
from functools import lru_cache
from langchain_core.documents import Document
import config

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a medical expert with deep knowledge of clinical research,
pharmacology, and biomedical literature. Answer the user's question using ONLY the
context provided below from peer-reviewed sources.
If the context is insufficient, state that clearly. Never fabricate clinical data.
Cite PMID or journal sources when available in the context."""

PROMPT_TEMPLATE = """{system}

Context:
{context}

Question: {question}

Answer:"""


def _get_llm():
    if config.EXPERT_BACKEND == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=config.GROQ_EXPERT_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.1,
            max_tokens=1024,
        )
    if config.EXPERT_BACKEND in ("hf", "transformers"):
        from src.agents.hf_transformers import invoke_text
        return lambda prompt: invoke_text(
            config.HF_MEDICAL_MODEL,
            prompt,
            max_new_tokens=1024,
            temperature=0.1,
        )
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=config.OLLAMA_MEDICAL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=1024,
    )


@lru_cache(maxsize=1)
def _get_retriever():
    from src.data_ingestion.embedder import load_medical_index
    from src.retrieval.hybrid_retriever import build_hybrid_retriever

    index = load_medical_index()
    import random
    all_docs = list(index.docstore._dict.values())
    bm25_docs = random.sample(all_docs, min(5000, len(all_docs)))
    return build_hybrid_retriever(bm25_docs, index, k=config.TOP_K)


def run(query: str, use_reranker: bool = True) -> dict:
    """
    Retrieve medical context and generate an answer.
    Returns dict with keys: answer, sources (list of Document dicts)
    """
    retriever = _get_retriever()
    raw_docs: list[Document] = retriever.invoke(query)

    if use_reranker:
        from src.retrieval.reranker import rerank
        raw_docs = rerank(query, raw_docs, top_k=config.TOP_K)

    context = "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('pmid', d.metadata.get('source','?'))} | {d.metadata.get('journal','?')}]\n{d.page_content}"
        for d in raw_docs
    )

    prompt = PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT, context=context, question=query
    )

    llm = _get_llm()
    if callable(llm):
        result = llm(prompt)
        answer = result.content if hasattr(result, "content") else str(result)
    else:
        result = llm.invoke(prompt)
        answer = result.content if hasattr(result, "content") else str(result)

    log.info("Medical expert generated answer (%d chars)", len(answer))
    return {
        "answer":  answer,
        "sources": [{"content": d.page_content, "metadata": d.metadata} for d in raw_docs],
    }
