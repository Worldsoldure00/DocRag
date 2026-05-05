"""
DocSight RAG — Streamlit Web UI
Run: streamlit run app/streamlit_app.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Force CPU and disable OpenMP thread pools.
# On Windows, PyTorch/OpenMP worker threads crash during teardown when
# model.encode() runs inside Streamlit's non-main worker thread.
# Setting these before any native library loads prevents thread pool creation.
os.environ["DOCSIGHT_FORCE_CPU"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st

# Cap torch to 1 inter-op and 1 intra-op thread — prevents OpenMP crash
# when PyTorch runs inside Streamlit's non-main worker thread on Windows.
try:
    from sentence_transformers import SentenceTransformer as _warmup  # noqa: F401
    import torch as _torch
    _torch.set_num_threads(1)
    _torch.set_num_interop_threads(1)
except Exception:
    pass

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocSight RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ DocSight RAG")
    st.markdown("**Multi-Agent Financial & Medical Q&A**")
    st.divider()

    import config
    backend = st.selectbox(
        "LLM Backend",
        ["groq", "ollama"],
        index=0 if config.LLM_BACKEND == "groq" else 1,
        help="groq = Groq API (fast, no local setup). ollama = local fine-tuned models.",
    )
    os.environ["LLM_BACKEND"] = backend
    config.LLM_BACKEND = backend

    use_reranker = st.toggle("Enable reranker", value=True,
                              help="Cross-encoder reranking (adds ~100ms, improves relevance)")

    st.divider()
    st.markdown("**Models in use**")
    st.caption(f"Router: `{config.GROQ_ROUTER_MODEL if backend == 'groq' else config.OLLAMA_ROUTER}`")
    st.caption(f"Expert: `{config.GROQ_EXPERT_MODEL if backend == 'groq' else 'fine-tuned Llama/BioMistral'}`")
    st.caption(f"Synth: `{config.GROQ_SYNTHESIZER_MODEL if backend == 'groq' else config.OLLAMA_SYNTH}`")
    st.caption(f"Finance Embed: `FinE5`")
    st.caption(f"Medical Embed: `PubMedBERT`")

    st.divider()
    st.markdown("**Example queries**")
    examples = [
        "What was Apple's net income in 2023?",
        "What are the side effects of metformin?",
        "How does GLP-1 drug revenue affect Eli Lilly's 2023 earnings?",
        "What is the mechanism of action of statins?",
        "Compare JPMorgan and Goldman Sachs R&D investment in 2023.",
        "What clinical trials are ongoing for Alzheimer's treatment?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["query_input"] = ex

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🔍 DocSight RAG")
st.markdown("Ask questions about **SEC filings** (finance) or **medical literature** — the system routes your query to the right expert automatically.")

query = st.text_area(
    "Your question",
    value=st.session_state.get("query_input", ""),
    height=80,
    placeholder="e.g. What was Microsoft's cloud revenue in 2023?",
    key="query_area",
)

col_run, col_clear = st.columns([1, 5])
with col_run:
    run_btn = st.button("Ask DocSight", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=False):
        st.session_state["query_input"] = ""
        st.rerun()

# ── Query execution ────────────────────────────────────────────────────────────
if run_btn and query.strip():
    with st.spinner("Routing query and retrieving sources..."):
        try:
            from src.agents.graph import run_query
            state = run_query(query.strip())
            # Force CUDA sync before leaving the heavy-compute section.
            # On Windows, CUDA tensor cleanup in Streamlit's worker thread
            # triggers an access violation if tensors are freed lazily.
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            with open("pipeline_error.log", "a") as _f:
                _f.write(err + "\n")
            st.markdown(f"**Pipeline error:** {e}")
            st.stop()

    domain     = state.get("domain", "?")
    answer     = state.get("final_answer", "No answer generated.")
    confidence = state.get("confidence", 0.0)
    sources    = state.get("all_sources", [])

    # ── Domain badge ──────────────────────────────────────────────────────────
    domain_colors = {"finance": "🟦", "medical": "🟩", "both": "🟨"}
    badge = domain_colors.get(domain, "⬜")
    st.markdown(f"### {badge} Domain: `{domain.upper()}`")

    # ── Confidence meter (plain markdown — avoids lazy-loaded JS chunks) ─────
    pct = round(confidence * 100, 1)
    bar_filled = int(pct / 5)  # 0-20 blocks
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    st.markdown(f"**Confidence:** `{pct}%`  `{bar}`")

    st.divider()

    # ── Answer ────────────────────────────────────────────────────────────────
    st.subheader("Answer")
    st.markdown(answer)

    st.divider()

    # ── Sources ───────────────────────────────────────────────────────────────
    if sources:
        import gc; gc.collect()
        st.subheader(f"Sources ({len(sources)} chunks retrieved)")

        finance_srcs = [s for s in sources if s["metadata"].get("domain") == "finance"]
        medical_srcs = [s for s in sources if s["metadata"].get("domain") == "medical"]

        tabs = []
        tab_names = []
        if finance_srcs:
            tab_names.append(f"📊 Finance ({len(finance_srcs)})")
        if medical_srcs:
            tab_names.append(f"🏥 Medical ({len(medical_srcs)})")
        if not tab_names:
            tab_names = ["All Sources"]
            finance_srcs = sources

        created_tabs = st.tabs(tab_names)

        def _render_sources(tab, src_list):
            with tab:
                for i, src in enumerate(src_list):
                    meta = src["metadata"]
                    if meta.get("domain") == "finance":
                        label = f"**[{i+1}]** {meta.get('ticker','?')} — {meta.get('filing_type','10-K')} | {meta.get('type','text')}"
                    else:
                        pmid = meta.get('pmid', '')
                        label = f"**[{i+1}]** PMID: {pmid} | {meta.get('journal','?')} {meta.get('year','')}"

                    with st.expander(label):
                        st.markdown(src["content"])

        idx = 0
        if finance_srcs:
            _render_sources(created_tabs[idx], finance_srcs)
            idx += 1
        if medical_srcs:
            _render_sources(created_tabs[idx], medical_srcs)
    else:
        st.markdown("_No source documents were retrieved._")

elif run_btn and not query.strip():
    st.markdown("**Please enter a question first.**")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("DocSight RAG | LangGraph + FAISS + FinE5 + PubMedBERT | CMSC 641 Final Project")
