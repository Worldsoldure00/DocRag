"""
DocSight RAG — Streamlit Web UI
Run: streamlit run app/streamlit_app.py
"""
import sys, os, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Thread / CPU guards ───────────────────────────────────────────────────────
os.environ["DOCSIGHT_FORCE_CPU"]      = "1"
os.environ["OMP_NUM_THREADS"]         = "1"
os.environ["MKL_NUM_THREADS"]         = "1"
os.environ["OPENBLAS_NUM_THREADS"]    = "1"
os.environ["NUMEXPR_NUM_THREADS"]     = "1"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Backend fixed to groq
import config
config.LLM_BACKEND = "groq"
os.environ["LLM_BACKEND"] = "groq"

try:
    import torch as _torch
    _torch.set_num_threads(1)
    _torch.set_num_interop_threads(1)
except Exception:
    pass

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Chat bubbles */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 12px 16px;
    margin-bottom: 8px;
}

/* Domain pill badges */
.badge-finance { background:#1565C0; color:#fff; padding:3px 10px;
                 border-radius:20px; font-size:11px; font-weight:600; }
.badge-medical { background: #3b185f; color: #d4b5ff; padding:3px 10px;
                 border-radius:20px; font-size:11px; font-weight:600; }
.badge-both    { background:#E65100; color:#fff; padding:3px 10px;
                 border-radius:20px; font-size:11px; font-weight:600; }
.badge-web     { background: #1a365d; color: #90cdf4; padding:3px 10px;
                 border-radius:20px; font-size:11px; font-weight:600; }

/* Source cards */
.src-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 3px solid rgba(255,255,255,0.25);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0 12px 0;
    font-size: 13px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ── Header row ────────────────────────────────────────────────────────────────
h_col, btn_col = st.columns([8, 1])
with h_col:
    st.markdown("# Multi-Agent RAG")
   # st.caption("Multi-agent Q&A over SEC filings & medical literature — powered by LangGraph + FAISS")
with btn_col:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

st.divider()

# ── Render conversation history ───────────────────────────────────────────────
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            meta = msg.get("meta", {})
            domain = meta.get("domain", "")
            if domain == "both":
                st.markdown('<span class="badge-finance">FINANCE</span> <span class="badge-medical">MEDICAL</span>', unsafe_allow_html=True)
            else:
                badge_cls = f"badge-{domain}" if domain in ("finance", "medical", "web") else ""
                if badge_cls:
                    st.markdown(f'<span class="{badge_cls}">{domain.upper()}</span>', unsafe_allow_html=True)
            pct = round(meta.get("confidence", 0) * 100, 1)
            filled = int(pct / 5)
            bar = "█" * filled + "░" * (20 - filled)
            st.caption(
                f"Confidence: {pct}%  `{bar}`  ·  "
                f"⏱ {meta.get('latency', 0):.1f}s"
            )
        st.markdown(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            sources = msg["sources"]
            fin_srcs = [s for s in sources if s["metadata"].get("domain") == "finance"]
            med_srcs = [s for s in sources if s["metadata"].get("domain") == "medical"]
            web_srcs = [s for s in sources if s["metadata"].get("domain") == "web"]
            all_groups = []
            if fin_srcs: all_groups.append(("Finance", fin_srcs))
            if med_srcs: all_groups.append(("Medical", med_srcs))
            if web_srcs: all_groups.append(("Web Search", web_srcs))
            if not all_groups: all_groups = [("Sources", sources)]

            with st.expander(f"Sources ({len(sources)} retrieved)"):
                for group_label, grp in all_groups:
                    st.markdown(f"###### {group_label}")
                    for i, src in enumerate(grp):
                        m = src["metadata"]
                        if m.get("domain") == "finance":
                            header = f"**[{i+1}]** `{m.get('source', 'Finance Document')}`"
                        elif m.get("domain") == "web":
                            header = f"**[{i+1}]** [{m.get('title', 'Web Source')}]({m.get('url', '#')})"
                        else:
                            title = m.get('title', 'Medical Document')
                            journal = m.get('journal', 'PubMed')
                            year = m.get('year', '')
                            header = f"**[{i+1}]** `{journal} {year}` — {title}"
                        st.markdown(header)
                        st.markdown(
                            f'<div class="src-card">{src["content"]}</div>',
                            unsafe_allow_html=True,
                        )

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question"):

    # Show user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run pipeline
    with st.chat_message("assistant"):
        t0 = time.time()
        state = {}
        with st.status("Initializing Pipeline...", expanded=True) as status_box:
            try:
                from src.agents.graph import stream_query
                
                # Iterate through the graph execution
                for step_dict in stream_query(prompt):
                    # step_dict format: {"node_name": AgentState}
                    for node, node_state in step_dict.items():
                        state = node_state  # Keep track of the latest state
                        
                        if node == "router":
                            status_box.update(label="Router Agents classifying domain...")
                            st.write(f"Classified query as: **{state.get('domain', 'unknown').upper()}**")
                        
                        elif node == "finance_agent":
                            status_box.update(label="Finance Expert searching SEC filings...")
                            st.write("Searching the Finance SEC Index...")
                            
                        elif node == "medical_agent":
                            status_box.update(label="Medical Expert searching PubMed documents...")
                            st.write("Searching the Medical PubMed Index...")
                            
                        elif node == "web_agent":
                            status_box.update(label="Local sources insufficient. Web Fallback scraping DuckDuckGo...")
                            st.write("Running live web search...")
                            
                        elif node == "synthesizer":
                            status_box.update(label="Synthesizer is drafting the final answer...")
                            st.write("Synthesizing context into final response...")

                # Done streaming
                status_box.update(label="Pipeline Complete", state="complete", expanded=False)

                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                elapsed = time.time() - t0
            except Exception as e:
                elapsed = time.time() - t0
                status_box.update(label="Pipeline Failed", state="error", expanded=True)
                st.error(f"Error: {e}")
                import traceback
                err_tb = traceback.format_exc()
                with open("pipeline_error.log", "a") as _f:
                    _f.write(err_tb + "\n")
                state = {
                    "domain": "?", "final_answer": f"**Pipeline error:** {e}",
                    "confidence": 0.0, "all_sources": []
                }

        domain     = state.get("domain", "?")
        answer     = state.get("final_answer", "No answer generated.")
        confidence = state.get("confidence", 0.0)
        sources    = state.get("all_sources", [])

        # Domain badge + confidence
        if domain == "both":
            st.markdown('<span class="badge-finance">FINANCE</span> <span class="badge-medical">MEDICAL</span>', unsafe_allow_html=True)
        else:
            badge_cls = f"badge-{domain}" if domain in ("finance", "medical", "web") else ""
            if badge_cls:
                st.markdown(f'<span class="{badge_cls}">{domain.upper()}</span>', unsafe_allow_html=True)
        pct = round(confidence * 100, 1)
        filled = int(pct / 5)
        bar = "█" * filled + "░" * (20 - filled)
        st.caption(f"Confidence: {pct}%  `{bar}`  ·  ⏱ {elapsed:.1f}")

        # Answer
        st.markdown(answer)

        # Sources — visible paragraphs
        if sources:
            fin_srcs = [s for s in sources if s["metadata"].get("domain") == "finance"]
            med_srcs = [s for s in sources if s["metadata"].get("domain") == "medical"]
            web_srcs = [s for s in sources if s["metadata"].get("domain") == "web"]
            all_groups = []
            if fin_srcs: all_groups.append(("Finance", fin_srcs))
            if med_srcs: all_groups.append(("Medical", med_srcs))
            if web_srcs: all_groups.append(("Web Search", web_srcs))
            if not all_groups: all_groups = [("Sources", sources)]

            with st.expander(f"Sources ({len(sources)} retrieved)"):
                for group_label, grp in all_groups:
                    st.markdown(f"###### {group_label}")
                    for i, src in enumerate(grp):
                        m = src["metadata"]
                        if m.get("domain") == "finance":
                            header = f"**[{i+1}]** `{m.get('source', 'Finance Document')}`"
                        elif m.get("domain") == "web":
                            header = f"**[{i+1}]** [{m.get('title', 'Web Source')}]({m.get('url', '#')})"
                        else:
                            title = m.get('title', 'Medical Document')
                            journal = m.get('journal', 'PubMed')
                            year = m.get('year', '')
                            header = f"**[{i+1}]** `{journal} {year}` — {title}"
                        st.markdown(header)
                        st.markdown(
                            f'<div class="src-card">{src["content"]}</div>',
                            unsafe_allow_html=True,
                        )

    # Persist to session
    meta = {"domain": domain, "confidence": confidence,
            "latency": elapsed, "n_sources": len(sources)}
    st.session_state["messages"].append({
        "role": "assistant", "content": answer,
        "meta": meta, "sources": sources,
    })
    st.rerun()

