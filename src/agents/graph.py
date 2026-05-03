"""
LangGraph stateful workflow — router → expert(s) → synthesizer.

State machine:
    router ──finance──► finance_agent ──► synthesizer
           ──medical──► medical_agent ──► synthesizer
           ──both────► finance_agent ──► medical_agent ──► synthesizer
"""
import logging
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
import config

log = logging.getLogger(__name__)


# ── State schema ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query:           str
    domain:          str                    # "finance" | "medical" | "both"
    finance_result:  Optional[dict]         # {answer, sources}
    medical_result:  Optional[dict]         # {answer, sources}
    final_answer:    str
    all_sources:     list[dict]
    confidence:      float
    error:           Optional[str]


# ── Node implementations ───────────────────────────────────────────────────────

def router_node(state: AgentState) -> AgentState:
    from src.agents.router import classify_domain
    try:
        domain = classify_domain(state["query"])
    except Exception as e:
        log.error("Router failed: %s", e)
        domain = "both"  # fallback to both on error
    log.info("Router → domain=%s", domain)
    return {**state, "domain": domain, "error": None}


def finance_node(state: AgentState) -> AgentState:
    from src.agents.finance_expert import run as finance_run
    try:
        result = finance_run(state["query"])
    except Exception as e:
        log.error("Finance agent failed: %s", e)
        result = {"answer": f"Finance agent error: {e}", "sources": []}
    return {**state, "finance_result": result}


def medical_node(state: AgentState) -> AgentState:
    from src.agents.medical_expert import run as medical_run
    try:
        result = medical_run(state["query"])
    except Exception as e:
        log.error("Medical agent failed: %s", e)
        result = {"answer": f"Medical agent error: {e}", "sources": []}
    return {**state, "medical_result": result}


def synthesizer_node(state: AgentState) -> AgentState:
    from src.agents.synthesizer import run as synth_run
    result = synth_run(
        query=state["query"],
        finance_result=state.get("finance_result"),
        medical_result=state.get("medical_result"),
    )
    return {
        **state,
        "final_answer": result["final_answer"],
        "confidence":   result["confidence"],
        "all_sources":  result["all_sources"],
    }


# ── Routing logic ──────────────────────────────────────────────────────────────

def _route_after_router(state: AgentState) -> str:
    domain = state.get("domain", "both")
    if domain == "finance":
        return "finance_agent"
    if domain == "medical":
        return "medical_agent"
    return "finance_agent"  # "both" → run finance first, medical second


def _route_after_finance(state: AgentState) -> str:
    if state.get("domain") == "both":
        return "medical_agent"
    return "synthesizer"


# ── Build & compile ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("router",        router_node)
    g.add_node("finance_agent", finance_node)
    g.add_node("medical_agent", medical_node)
    g.add_node("synthesizer",   synthesizer_node)

    g.set_entry_point("router")

    g.add_conditional_edges("router", _route_after_router, {
        "finance_agent": "finance_agent",
        "medical_agent": "medical_agent",
    })

    g.add_conditional_edges("finance_agent", _route_after_finance, {
        "medical_agent": "medical_agent",
        "synthesizer":   "synthesizer",
    })

    g.add_edge("medical_agent", "synthesizer")
    g.add_edge("synthesizer", END)

    return g


# Compiled app — import this in the Streamlit app
_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_graph().compile()
    return _app


def run_query(query: str) -> AgentState:
    """Convenience wrapper — run a single query through the full graph."""
    app = get_app()
    initial_state: AgentState = {
        "query":          query,
        "domain":         "",
        "finance_result": None,
        "medical_result": None,
        "final_answer":   "",
        "all_sources":    [],
        "confidence":     0.0,
        "error":          None,
    }
    return app.invoke(initial_state)
