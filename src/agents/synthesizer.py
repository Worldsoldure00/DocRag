"""
Synthesizer agent — merges answers from domain experts into a single coherent
response with citations and a confidence score.
No fine-tuning needed — uses base Llama-3.3-70B via Groq or Ollama.
"""
import logging
import config

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a synthesis assistant. Your job is to combine domain-expert answers
into one coherent, well-structured response for the user.

Rules:
1. Preserve all factual claims and citations from the expert answers.
2. Remove redundancy but do not lose information.
3. Structure: brief direct answer first, then supporting details.
4. End with a one-line confidence statement: "Confidence: X/10 — [reason]"
5. If only one domain answer is present, just polish and cite it."""

PROMPT_TEMPLATE = """{system}

{expert_answers}

Original question: {question}

Synthesized answer:"""


def _get_llm():
    if config.LLM_BACKEND == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=config.GROQ_SYNTHESIZER_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.2,
            max_tokens=1500,
        )
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=config.OLLAMA_SYNTH,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.2,
        num_predict=1500,
    )


def _compute_confidence(finance_result: dict | None, medical_result: dict | None) -> float:
    """Heuristic confidence 0–1 based on source coverage."""
    n_sources = 0
    if finance_result:
        n_sources += len(finance_result.get("sources", []))
    if medical_result:
        n_sources += len(medical_result.get("sources", []))
    import random
    if n_sources >= 5:
        # User requested variance between 85% and 95%
        return random.uniform(0.85, 0.95)
    
    # Otherwise scale based on sources with a slight random jitter
    base = 0.3 + (0.12 * n_sources)
    jitter = random.uniform(-0.03, 0.04)
    return min(max(base + jitter, 0.1), 0.95)


def run(
    query: str,
    finance_result: dict | None = None,
    medical_result: dict | None = None,
) -> dict:
    """
    Synthesize one or two domain answers into a final response.
    Returns dict with: final_answer, confidence, all_sources
    """
    parts = []
    if finance_result and finance_result.get("answer"):
        parts.append(f"[Financial Analysis]\n{finance_result['answer']}")
    if medical_result and medical_result.get("answer"):
        parts.append(f"[Medical Analysis]\n{medical_result['answer']}")

    if not parts:
        return {
            "final_answer": "No relevant information found in either domain.",
            "confidence":   0.1,
            "all_sources":  [],
        }

    # If only one domain, skip the LLM synthesis call — just return directly
    if len(parts) == 1:
        answer = parts[0].split("\n", 1)[1] if "\n" in parts[0] else parts[0]
    else:
        expert_block = "\n\n".join(parts)
        prompt = PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            expert_answers=expert_block,
            question=query,
        )
        llm    = _get_llm()
        result = llm.invoke(prompt)
        answer = result.content if hasattr(result, "content") else str(result)

    all_sources = []
    if finance_result:
        all_sources.extend(finance_result.get("sources", []))
    if medical_result:
        all_sources.extend(medical_result.get("sources", []))

    confidence = _compute_confidence(finance_result, medical_result)
    log.info("Synthesizer done. confidence=%.2f sources=%d", confidence, len(all_sources))

    return {
        "final_answer": answer,
        "confidence":   confidence,
        "all_sources":  all_sources,
    }
