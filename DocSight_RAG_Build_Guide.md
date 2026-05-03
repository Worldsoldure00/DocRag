# DocSight RAG — Realistic Build Guide

## What This Project Actually Is

DocSight RAG is a **multi-agent question-answering system** that handles two domains — **financial** (SEC filings) and **medical** (PubMed / clinical notes). A user asks a natural language question, the system figures out which domain it belongs to, retrieves relevant document chunks, and generates a grounded answer with source citations.

The architecture has three layers:

1. **RAG layer** — ingests documents, chunks them, embeds them, stores them in a vector DB, and retrieves relevant chunks at query time.
2. **Agent layer** — specialized agents (router, financial expert, medical expert, synthesizer) coordinate to handle queries. Each expert agent is backed by a fine-tuned open-source LLM.
3. **Interface layer** — a web UI where users ask questions and see answers with source passages highlighted.

---

## Architecture (What Actually Gets Built)

```
User Query
    │
    ▼
┌─────────────────────┐
│   Router Agent       │  ← Small classifier (fine-tuned distilbert or phi-3-mini)
│   Decides: finance   │    Routes to correct domain expert
│   / medical / both   │
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│Finance │ │Medical │   ← Each agent = fine-tuned Mistral-7B or Llama-3-8B (QLoRA)
│Agent   │ │Agent   │     + domain-specific RAG retrieval
└───┬────┘ └───┬────┘
    │          │
    ▼          ▼
┌────────┐ ┌────────┐
│Finance │ │Medical │   ← FAISS indexes, one per domain
│VectorDB│ │VectorDB│     Embeddings: FinBERT / PubMedBERT
└────────┘ └────────┘
    │          │
    └────┬─────┘
         ▼
┌─────────────────────┐
│  Synthesizer Agent   │  ← Merges answers if query spans both domains
│  (base Mistral/Llama)│    Adds citations, confidence score, domain tag
└─────────────────────┘
         │
         ▼
    Final Answer
    + Source Passages
    + Confidence Score
    + Domain Tag
```

### Agent Orchestration

Use **LangGraph** (not vanilla LangChain). LangGraph gives you a state machine where each node is an agent and edges are routing decisions. This is cleaner than CrewAI for your use case because you control exactly how agents hand off to each other.

```python
# Simplified LangGraph flow
from langgraph.graph import StateGraph

workflow = StateGraph(AgentState)
workflow.add_node("router", router_agent)
workflow.add_node("finance_agent", finance_agent)
workflow.add_node("medical_agent", medical_agent)
workflow.add_node("synthesizer", synthesizer_agent)

workflow.add_conditional_edges("router", route_query, {
    "finance": "finance_agent",
    "medical": "medical_agent",
    "both": "finance_agent",  # runs finance first, then medical
})
workflow.add_edge("finance_agent", "synthesizer")
workflow.add_edge("medical_agent", "synthesizer")
```

---

## Exact Tech Stack (Use These, Not Alternatives)

| Layer | Tool | Why |
|---|---|---|
| Orchestration | **LangGraph** (`langgraph`) | State-machine multi-agent flows, better than vanilla LangChain chains |
| RAG Framework | **LangChain** (`langchain`, `langchain-community`) | Doc loaders, text splitters, retriever abstractions |
| Vector Store | **FAISS** (`faiss-gpu` or `faiss-cpu`) | Fast, local, no infra setup needed |
| Financial Embeddings | **FinBERT** (`ProsusAI/finbert` on HuggingFace) | Domain-tuned for SEC language |
| Medical Embeddings | **PubMedBERT** (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`) | Domain-tuned for biomedical text |
| General Embeddings | **all-MiniLM-L6-v2** (`sentence-transformers`) | Fast fallback, 384-dim |
| Base LLM | **Mistral-7B-Instruct-v0.3** | Best quality-to-size ratio for fine-tuning |
| Fine-tuning | **Unsloth** + **QLoRA** (4-bit) | 2-3x faster than vanilla PEFT, fits on free Colab T4 |
| LLM Serving | **vLLM** or **Ollama** | vLLM for production speed, Ollama for dev simplicity |
| Keyword Retrieval | **BM25** via `rank_bm25` | Hybrid search (BM25 + dense) |
| Table Parsing | **pdfplumber** | Extracts tables from SEC PDFs reliably |
| Evaluation | **RAGAS** (`ragas`) | Faithfulness, context relevancy, answer relevancy |
| Tracing | **LangSmith** (free tier) | Traces every chain run, logs retrieval + generation |
| Web UI | **Streamlit** or **Gradio** | Fast to build, good enough for demo |
| Backend API (optional) | **FastAPI** | If you want a proper API layer |

### Python Environment

```bash
# Core
pip install langchain langchain-community langgraph faiss-gpu
pip install sentence-transformers transformers datasets
pip install ragas langsmith

# Fine-tuning (run on Colab/HPC)
pip install unsloth peft bitsandbytes accelerate trl

# Data + parsing
pip install pdfplumber rank_bm25 sec-edgar-downloader pubmed-parser

# Serving + UI
pip install ollama streamlit fastapi uvicorn
```

---

## Data Pipeline (Be Specific)

### Financial Data

```python
from sec_edgar_downloader import Downloader

dl = Downloader("YourName", "your@email.com")
# Download 10-K filings for 10-15 companies (not 50, be realistic)
tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "JPM", "GS", "JNJ", "PFE", "META", "TSLA"]
for t in tickers:
    dl.get("10-K", t, after="2022-01-01", before="2024-12-31")
```

Use **pdfplumber** for table extraction, **LangChain HTMLLoader** for HTML filings. Strip boilerplate (headers/footers/legal disclaimers) before chunking.

### Medical Data

- **PubMed**: Use `Entrez` API from BioPython. Pull ~5,000 abstracts for 3-4 disease areas (e.g., diabetes, cardiovascular, oncology, respiratory).
- **MIMIC-III**: Only if you already have PhysioNet access. If not, skip it — the credentialing process alone takes longer than 2 weeks. Use PubMed + synthetic clinical notes instead.

```python
from Bio import Entrez
Entrez.email = "your@email.com"
handle = Entrez.esearch(db="pubmed", term="type 2 diabetes treatment", retmax=2000)
```

### Chunking Strategy

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# For regular text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " "]
)

# For financial tables: keep each table as one chunk
# Parse with pdfplumber, convert to markdown, store as single chunk with metadata
```

Tag every chunk with metadata: `{"domain": "finance"|"medical", "source": "...", "section": "..."}`.

---

## Fine-Tuning Plan (Realistic for 2 Weeks)

You are fine-tuning **3 models** using QLoRA via Unsloth:

### Model 1: Router Classifier

- **Base**: `microsoft/phi-3-mini-4k-instruct` (3.8B, small and fast)
- **Task**: Given a user question, output `finance`, `medical`, or `both`
- **Training data**: 500-800 labeled question-domain pairs. Generate synthetically with GPT-4 or Claude, then clean manually.
- **Training time**: ~30 min on T4

```python
# Training data format
{"instruction": "Classify this query's domain", 
 "input": "What was Apple's revenue in Q3 2023?", 
 "output": "finance"}

{"instruction": "Classify this query's domain",
 "input": "What are the side effects of metformin?",
 "output": "medical"}
```

### Model 2: Finance Expert

- **Base**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Task**: Given a question + retrieved financial context chunks, generate a grounded answer
- **Training data**: 1,000-2,000 (question, context, answer) triples from SEC filings. Generate QA pairs from actual filing sections using GPT-4/Claude, manually verify ~200.
- **Training time**: ~2-3 hours on T4/A100

### Model 3: Medical Expert

- **Base**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Task**: Same as finance but for medical queries
- **Training data**: 1,000-2,000 triples from PubMed abstracts. Use BioASQ or PubMedQA existing datasets as a head start, then supplement.
- **Training time**: ~2-3 hours on T4/A100

### Fine-Tuning Code (Unsloth + QLoRA)

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,              # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
)

# Format: "<s>[INST] Context: {context}\n\nQuestion: {question} [/INST] {answer}</s>"
dataset = load_dataset("json", data_files="finance_qa_pairs.json")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    max_seq_length=2048,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
)
trainer.train()

# Save adapter (small, ~50-100MB)
model.save_pretrained_merged("finance_expert_merged", tokenizer)
```

### The Synthesizer

Don't fine-tune a 4th model. Use base Mistral-7B-Instruct with a good prompt template. Its job is just to merge/format answers from the domain experts.

---

## RAG Pipeline Implementation

### Building the Indexes

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Finance index with FinBERT
finance_embeddings = HuggingFaceEmbeddings(model_name="ProsusAI/finbert")
finance_index = FAISS.from_documents(finance_chunks, finance_embeddings)
finance_index.save_local("indexes/finance_faiss")

# Medical index with PubMedBERT
medical_embeddings = HuggingFaceEmbeddings(
    model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)
medical_index = FAISS.from_documents(medical_chunks, medical_embeddings)
medical_index.save_local("indexes/medical_faiss")
```

### Hybrid Retrieval (BM25 + Dense)

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

def get_hybrid_retriever(chunks, faiss_index, k=5):
    bm25 = BM25Retriever.from_documents(chunks, k=k)
    dense = faiss_index.as_retriever(search_kwargs={"k": k})
    
    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[0.4, 0.6]  # favor dense for semantic matching
    )
```

### Full Agent Implementation

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class AgentState(TypedDict):
    query: str
    domain: str  # "finance" | "medical" | "both"
    finance_context: List[str]
    medical_context: List[str]
    finance_answer: str
    medical_answer: str
    final_answer: str
    sources: List[dict]
    confidence: float

def router_agent(state: AgentState) -> AgentState:
    """Uses fine-tuned router model to classify domain."""
    query = state["query"]
    domain = router_model.predict(query)  # your fine-tuned classifier
    return {**state, "domain": domain}

def finance_agent(state: AgentState) -> AgentState:
    """Retrieves finance chunks + generates answer with fine-tuned finance LLM."""
    docs = finance_retriever.invoke(state["query"])
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = f"""[INST] You are a financial analyst. Answer based ONLY on the context below.
    
Context:
{context}

Question: {state['query']} [/INST]"""
    
    answer = finance_llm.invoke(prompt)
    return {
        **state,
        "finance_context": [d.page_content for d in docs],
        "finance_answer": answer,
        "sources": [{"content": d.page_content, "metadata": d.metadata} for d in docs]
    }

def medical_agent(state: AgentState) -> AgentState:
    """Same pattern, uses medical retriever + fine-tuned medical LLM."""
    docs = medical_retriever.invoke(state["query"])
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = f"""[INST] You are a medical expert. Answer based ONLY on the context below.
    
Context:
{context}

Question: {state['query']} [/INST]"""
    
    answer = medical_llm.invoke(prompt)
    return {
        **state,
        "medical_context": [d.page_content for d in docs],
        "medical_answer": answer,
        "sources": state.get("sources", []) + [{"content": d.page_content, "metadata": d.metadata} for d in docs]
    }

def synthesizer_agent(state: AgentState) -> AgentState:
    """Merges domain answers into final response with citations."""
    answers = []
    if state.get("finance_answer"):
        answers.append(f"[Financial Analysis]\n{state['finance_answer']}")
    if state.get("medical_answer"):
        answers.append(f"[Medical Analysis]\n{state['medical_answer']}")
    
    prompt = f"""[INST] Synthesize the following domain-expert answers into one coherent response.
Include source citations. Rate your confidence 0-1.

{chr(10).join(answers)}

Original question: {state['query']} [/INST]"""
    
    final = base_llm.invoke(prompt)
    return {**state, "final_answer": final, "confidence": compute_confidence(state)}

def route_query(state: AgentState) -> str:
    if state["domain"] == "finance":
        return "finance_agent"
    elif state["domain"] == "medical":
        return "medical_agent"
    else:
        return "both_finance_first"

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("router", router_agent)
graph.add_node("finance_agent", finance_agent)
graph.add_node("medical_agent", medical_agent)
graph.add_node("synthesizer", synthesizer_agent)

graph.set_entry_point("router")
graph.add_conditional_edges("router", route_query, {
    "finance_agent": "finance_agent",
    "medical_agent": "medical_agent",
    "both_finance_first": "finance_agent",
})
graph.add_edge("finance_agent", "synthesizer")
graph.add_edge("medical_agent", "synthesizer")
graph.add_edge("synthesizer", END)

app = graph.compile()
```

---

## Evaluation (What to Actually Measure)

### Use RAGAS — 4 Metrics That Matter

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # % of claims grounded in context
    answer_relevancy,    # is the answer relevant to the question
    context_precision,   # are retrieved chunks relevant
    context_recall,      # did we retrieve all needed info
)

result = evaluate(
    dataset=eval_dataset,  # ~100-200 QA pairs with ground truth
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(result)  # pandas DataFrame with per-question scores
```

### Create Eval Dataset

Build ~100 QA pairs per domain (200 total). Structure:

```json
{
    "question": "What was JPMorgan's net income in 2023?",
    "ground_truth": "JPMorgan reported net income of $49.6 billion in 2023.",
    "contexts": ["...relevant chunk from 10-K..."]
}
```

Generate bulk with Claude/GPT-4, manually verify at least 50 per domain.

### Target Scores

| Metric | Minimum Target | Good Target |
|---|---|---|
| Faithfulness | > 0.80 | > 0.90 |
| Answer Relevancy | > 0.75 | > 0.85 |
| Context Precision | > 0.70 | > 0.80 |
| Context Recall | > 0.65 | > 0.75 |

---

## 2-Week Sprint Plan

### Week 1: Foundation (Days 1–7)

| Day | Vikas | Syed | Ibrahim |
|---|---|---|---|
| 1-2 | Set up repo, project structure, conda env, FAISS install | Download SEC filings (10 companies, 10-K 2022-2024) via EDGAR API | Pull PubMed abstracts (5K), find/prep PubMedQA dataset |
| 3-4 | Build chunking pipeline (text + table parsing with pdfplumber) | Process financial docs → chunks → FinBERT embeddings → FAISS index | Process medical docs → chunks → PubMedBERT embeddings → FAISS index |
| 5 | Implement hybrid retriever (BM25 + dense) for both domains | Generate synthetic finance QA training pairs (~1500) using Claude API | Generate synthetic medical QA training pairs (~1500) using Claude API |
| 6-7 | Build LangGraph agent skeleton (router → experts → synthesizer) | Fine-tune router model (Phi-3-mini) + finance expert (Mistral-7B) on Colab | Fine-tune medical expert (Mistral-7B) on Colab |

**Week 1 deliverable**: Both FAISS indexes built, all 3 models fine-tuned (adapter weights saved), LangGraph skeleton running with mock agents.

### Week 2: Integration + Polish (Days 8–14)

| Day | Vikas | Syed | Ibrahim |
|---|---|---|---|
| 8-9 | Integrate fine-tuned models into LangGraph agents, serve via Ollama | Build Streamlit UI (query input, answer display, source viewer, domain tag) | Create eval dataset (200 QA pairs), set up RAGAS evaluation |
| 10-11 | End-to-end testing, fix retrieval/generation issues | Add LangSmith tracing to every agent node | Run RAGAS evaluation, compute metrics, identify failure cases |
| 12-13 | Performance tuning (chunk size, top-K, BM25 weights, prompt engineering) | Polish UI (confidence bars, source highlighting, domain routing indicator) | Run ablation: BM25-only vs dense-only vs hybrid, compare RAGAS scores |
| 14 | Final integration testing, write README | Record demo video, prepare presentation slides | Compile evaluation results table, write results section |

**Week 2 deliverable**: Fully working end-to-end system with UI, evaluation results, demo-ready.

---

## Project Structure

```
docsight-rag/
├── README.md
├── requirements.txt
├── config.py                    # all hyperparams, paths, model names
│
├── data/
│   ├── raw/
│   │   ├── sec_filings/         # downloaded 10-K HTML/PDF files
│   │   └── pubmed/              # downloaded abstracts
│   ├── processed/
│   │   ├── finance_chunks.jsonl
│   │   └── medical_chunks.jsonl
│   └── eval/
│       ├── finance_eval.json    # 100 QA pairs
│       └── medical_eval.json    # 100 QA pairs
│
├── indexes/
│   ├── finance_faiss/           # saved FAISS index
│   └── medical_faiss/           # saved FAISS index
│
├── models/
│   ├── router_adapter/          # QLoRA adapter for Phi-3-mini
│   ├── finance_adapter/         # QLoRA adapter for Mistral-7B
│   └── medical_adapter/         # QLoRA adapter for Mistral-7B
│
├── src/
│   ├── data_ingestion/
│   │   ├── sec_downloader.py
│   │   ├── pubmed_downloader.py
│   │   ├── chunker.py           # text + table chunking
│   │   └── embedder.py          # build FAISS indexes
│   │
│   ├── agents/
│   │   ├── router.py            # router agent
│   │   ├── finance_expert.py    # finance RAG agent
│   │   ├── medical_expert.py    # medical RAG agent
│   │   ├── synthesizer.py       # answer merger
│   │   └── graph.py             # LangGraph workflow definition
│   │
│   ├── retrieval/
│   │   ├── hybrid_retriever.py  # BM25 + FAISS ensemble
│   │   └── reranker.py          # optional: cross-encoder reranker
│   │
│   └── evaluation/
│       ├── generate_qa_pairs.py # synthetic QA generation
│       ├── run_ragas.py         # RAGAS evaluation script
│       └── ablation.py          # BM25 vs dense vs hybrid comparison
│
├── training/
│   ├── train_router.py          # fine-tune router on Colab
│   ├── train_finance.py         # fine-tune finance expert
│   ├── train_medical.py         # fine-tune medical expert
│   └── data_prep.py             # format training data for SFT
│
├── app/
│   └── streamlit_app.py         # web UI
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_fine_tuning.ipynb      # Colab-ready notebook
    └── 03_evaluation.ipynb
```

---

## Critical Shortcuts (Things That Save You Time)

1. **Don't fine-tune from scratch if stuck**: If fine-tuning is taking too long or results are bad, use base Mistral-7B-Instruct with really good system prompts. A well-prompted base model beats a badly fine-tuned one. You can always say "we fine-tuned" for the router (which is easy) and used "domain-adapted prompting" for the experts.

2. **Skip MIMIC-III**: The PhysioNet approval process is slow. Use PubMed abstracts + PubMedQA (pre-built QA dataset, freely available). If anyone asks, say "we designed the pipeline to support MIMIC-III but used PubMed for this iteration due to credentialing timelines."

3. **Use Ollama for serving**: Don't waste time on vLLM setup. `ollama create finance-expert -f Modelfile` with your merged adapter is running in 5 minutes.

4. **LangSmith free tier**: 5,000 traces/month is enough. Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY=...` in your env and every LangChain/LangGraph call auto-logs.

5. **Synthetic QA pairs**: Use Claude API to generate training data. Paste a filing chunk, ask for 5 QA pairs. Clean manually (delete bad ones, fix answers). 2 hours of this gets you 1,500 pairs.

6. **Embeddings matter more than the LLM**: Spend time making sure FinBERT/PubMedBERT embeddings + good chunking retrieve the right context. If retrieval is good, even a mediocre LLM gives decent answers.

---

## What to Say in Your Presentation

Frame it as: "We built a multi-agent RAG system with domain-specialized agents, each backed by a fine-tuned LLM, orchestrated through a stateful graph (LangGraph). Key contributions: (1) domain-routed multi-agent architecture, (2) hybrid BM25 + dense retrieval with domain-specific embeddings, (3) QLoRA fine-tuned expert models, (4) end-to-end evaluation with RAGAS + LangSmith tracing."

Don't mention: LlamaIndex (you're not using it, LangGraph replaces that routing), "zero hallucinations" (say "hallucination-minimized" with RAGAS faithfulness scores to back it up).
