"""
Router agent — classifies an incoming query as 'finance', 'medical', or 'both'.
Uses Groq (fast, hosted) or Ollama (local fine-tuned Phi-4-mini).
"""
import re
import logging
import config

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an intelligent query router for a document Q&A system.
Given a user question, classify it into exactly ONE of these domains:
- finance   → purely financial topics (companies, stocks, SEC filings, earnings, revenue).
- medical   → purely medical topics (diseases, drugs, treatments, clinical trials, symptoms).
- both      → the question contains BOTH financial concepts (e.g. cost, impact, market, finance) AND medical concepts (e.g. clinical, severe hypertension, treatment). If you see a disease/treatment AND a financial term like 'cost' or 'market impact', you MUST choose 'both'.

Respond with ONLY one word: finance, medical, or both. No explanation."""


def _parse_domain(raw: str) -> str:
    raw = raw.strip().lower()
    for label in ("both", "finance", "medical"):
        if label in raw:
            return label
    return "finance"  # safe fallback


def classify_domain(query: str) -> str:
    """Return 'finance', 'medical', or 'both' for the given query."""
    if config.ROUTER_BACKEND == "groq":
        return _classify_groq(query)
    if config.ROUTER_BACKEND in ("hf", "transformers"):
        return _classify_hf(query)
    return _classify_ollama(query)


def _classify_groq(query: str) -> str:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatGroq(
        model=config.GROQ_ROUTER_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0,
        max_tokens=10,
    )
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query),
    ])
    domain = _parse_domain(response.content)
    log.debug("Router (Groq) → %s for query: %s", domain, query[:80])
    return domain


def _classify_ollama(query: str) -> str:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatOllama(
        model=config.OLLAMA_ROUTER,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0,
        num_predict=10,
    )
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query),
    ])
    domain = _parse_domain(response.content)
    log.debug("Router (Ollama) → %s for query: %s", domain, query[:80])
    return domain


def _classify_hf(query: str) -> str:
    from src.agents.hf_transformers import generate_text

    prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {query}\n\nAnswer:"
    raw = generate_text(
        config.HF_ROUTER_MODEL,
        prompt,
        max_new_tokens=10,
        temperature=0.0,
    )
    domain = _parse_domain(raw)
    log.debug("Router (HF) → %s for query: %s", domain, query[:80])
    return domain
