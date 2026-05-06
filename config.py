import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
INDEXES_DIR = BASE_DIR / "indexes"
MODELS_DIR  = BASE_DIR / "models"

RAW_SEC_DIR    = DATA_DIR / "raw" / "sec_filings"
RAW_PUBMED_DIR = DATA_DIR / "raw" / "pubmed"
PROCESSED_DIR  = DATA_DIR / "processed"
EVAL_DIR       = DATA_DIR / "eval"

FINANCE_INDEX_PATH = INDEXES_DIR / "finance_faiss"
MEDICAL_INDEX_PATH = INDEXES_DIR / "medical_faiss"

FINANCE_CHUNKS_PATH = PROCESSED_DIR / "finance_chunks.jsonl"
MEDICAL_CHUNKS_PATH = PROCESSED_DIR / "medical_chunks.jsonl"

# ── API Keys ──────────────────────────────────────────────────────────────────
HF_TOKEN          = os.getenv("HF_TOKEN")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

ENTREZ_EMAIL    = os.getenv("ENTREZ_EMAIL", "placeholder@email.com")
SEC_EDGAR_NAME  = os.getenv("SEC_EDGAR_NAME", "Multi-Agent RAG")
SEC_EDGAR_EMAIL = os.getenv("SEC_EDGAR_EMAIL", "placeholder@email.com")

# Enable LangSmith tracing only when key is present
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]    = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "docsight-rag")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# ── LLM Serving ───────────────────────────────────────────────────────────────
# "groq"   → uses Groq API with base models (works immediately, no fine-tuning needed)
# "ollama" → uses locally served fine-tuned GGUF models via Ollama
LLM_BACKEND    = os.getenv("LLM_BACKEND", "groq")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── HuggingFace Model IDs ─────────────────────────────────────────────────────
ROUTER_BASE_MODEL  = "microsoft/phi-4-mini-instruct"
FINANCE_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MEDICAL_BASE_MODEL = "BioMistral/BioMistral-7B"

FINANCE_EMBED_MODEL = "BAAI/bge-large-en-v1.5"          # 1024-dim, MTEB top-10, strong on financial text
MEDICAL_EMBED_MODEL = "NeuML/pubmedbert-base-embeddings" # 768-dim, purpose-built for PubMed retrieval

# ── Groq Models ───────────────────────────────────────────────────────────────
# Fast hosted inference — used when LLM_BACKEND=groq or for QA generation
GROQ_ROUTER_MODEL     = "llama-3.1-8b-instant"      # fast routing
GROQ_EXPERT_MODEL     = "llama-3.3-70b-versatile"   # best quality for expert answers
GROQ_SYNTHESIZER_MODEL = "llama-3.3-70b-versatile"
GROQ_QA_GEN_MODEL     = "llama-3.3-70b-versatile"   # synthetic QA pair generation

# ── Ollama Model Names (after fine-tuning + GGUF conversion) ──────────────────
OLLAMA_ROUTER    = "phi4-mini-router"
OLLAMA_FINANCE   = "llama31-finance-expert"
OLLAMA_MEDICAL   = "biomistral-medical-expert"
OLLAMA_SYNTH     = "llama31-finance-expert"  # reuse finance model as synthesizer base

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K        = 5
BM25_WEIGHT  = 0.4
DENSE_WEIGHT = 0.6

# ── Fine-tuning (QLoRA) ───────────────────────────────────────────────────────
LORA_RANK           = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
MAX_SEQ_LENGTH      = 2048
TRAIN_BATCH_SIZE    = 4
GRAD_ACCUMULATION   = 4
LEARNING_RATE       = 2e-4
NUM_EPOCHS          = 3
WARMUP_STEPS        = 50

# ── Data Collection ───────────────────────────────────────────────────────────
SEC_TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "JPM",
    "GS",   "JNJ",  "PFE",  "META", "TSLA",
]
SEC_FILING_TYPE    = "10-K"
SEC_DATE_AFTER     = "2022-01-01"
SEC_DATE_BEFORE    = "2024-12-31"

PUBMED_QUERIES = [
    "type 2 diabetes treatment clinical trial",
    "cardiovascular disease risk factors management",
    "lung cancer immunotherapy outcomes",
    "hypertension antihypertensive drug therapy",
]
PUBMED_MAX_RESULTS = 2000   # per query → ~8K abstracts total

# ── Index size cap (subsample for demo; set to None for full corpus) ──────────
# Finance has 871K chunks — cap at 60K for manageable local indexing
# Increase or set None after moving to a GPU cluster
MAX_FINANCE_CHUNKS = 60_000
MAX_MEDICAL_CHUNKS = None   # 50K medical chunks — keep all

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_SAMPLE_SIZE      = 100   # QA pairs per domain
EVAL_FINANCE_PATH     = EVAL_DIR / "finance_eval.json"
EVAL_MEDICAL_PATH     = EVAL_DIR / "medical_eval.json"
RAGAS_LLM_MODEL       = GROQ_EXPERT_MODEL
