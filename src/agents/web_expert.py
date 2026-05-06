"""
Web expert agent — retrieves web search results via DuckDuckGo and generates a
grounded answer when local databases lack relevant documents.
"""
import logging
import config

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful research assistant. Answer the user's question using ONLY the
web search results provided below.
If the web search results do not contain enough information, say so explicitly.
Always cite the source links provided in the context.
DO NOT append generic disclaimers (e.g., "Further research is necessary..."). Give direct answers only."""

PROMPT_TEMPLATE = """{system}

Web Search Results:
{context}

Question: {question}

Answer:"""


def _get_llm():
    if config.WEB_BACKEND == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=config.GROQ_EXPERT_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.2,
            max_tokens=1024,
        )
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=config.OLLAMA_MEDICAL,  # fallback to the medical/general model
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.2,
        num_predict=1024,
    )


def run(query: str) -> dict:
    """
    Search the web and generate an answer.
    Returns dict with keys: answer, sources (list of dicts)
    """
    log.info("Running web expert fallback for query: '%s'", query)
    
    try:
        from ddgs import DDGS
        raw_results = DDGS().text(query, max_results=5)
    except Exception as e:
        log.error("DuckDuckGo search failed: %s", e)
        raw_results = []

    if not raw_results:
        return {
            "answer": "No relevant web search results found.",
            "sources": []
        }

    # Format context for the LLM
    context_parts = []
    sources = []
    
    for i, res in enumerate(raw_results):
        snippet = res.get("body", "")
        title = res.get("title", "Web Source")
        link = res.get("href", "")
        
        context_parts.append(f"[Source {i+1}: {title} | URL: {link}]\n{snippet}")
        
        sources.append({
            "content": snippet,
            "metadata": {
                "domain": "web",
                "title": title,
                "url": link,
                "source": "duckduckgo"
            }
        })
        
    context = "\n\n---\n\n".join(context_parts)

    prompt = PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT, context=context, question=query
    )

    llm    = _get_llm()
    result = llm.invoke(prompt)
    answer = result.content if hasattr(result, "content") else str(result)

    log.info("Web expert generated answer (%d chars)", len(answer))
    return {
        "answer":  answer,
        "sources": sources,
    }
