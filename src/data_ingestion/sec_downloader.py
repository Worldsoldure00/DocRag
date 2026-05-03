"""
Finance data ingestion — two sources combined:
  1. Already-downloaded EDGAR full-submission.txt files (data/raw/sec_filings/)
  2. HuggingFace: virattt/financial-qa-10K  (4K QA pairs for training)
                  PatronusAI/financebench    (eval ground truth)

Run: python -m src.data_ingestion.sec_downloader
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import re
import logging
from pathlib import Path

from bs4 import BeautifulSoup
from datasets import load_dataset

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BOILERPLATE = [
    "table of contents", "forward-looking statements", "incorporated by reference",
    "exhibit index", "part i", "part ii", "part iii", "part iv",
]


# ── Parse EDGAR full-submission.txt ──────────────────────────────────────────

def _extract_html_from_submission(path: Path) -> list[str]:
    """
    EDGAR full-submission.txt embeds HTML inside <TEXT>...</TEXT> blocks.
    Extract and parse each block.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        log.warning("Cannot read %s: %s", path, e)
        return []

    # Find all <TEXT> blocks (there may be multiple documents per submission)
    blocks = re.findall(r"<TEXT>(.*?)</TEXT>", raw, re.DOTALL | re.IGNORECASE)
    texts = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # If it looks like HTML, parse it
        if "<html" in block.lower() or "<body" in block.lower() or "<p" in block.lower():
            soup = BeautifulSoup(block, "lxml")
            for tag in soup(["script", "style", "header", "footer", "nav", "table"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        else:
            text = block  # plain text block

        # Strip boilerplate lines
        clean_lines = []
        for line in text.splitlines():
            if len(line.strip()) < 10:
                continue
            if any(bp in line.lower() for bp in BOILERPLATE):
                continue
            clean_lines.append(line)

        cleaned = "\n".join(clean_lines).strip()
        if len(cleaned) > 200:
            texts.append(cleaned)

    return texts


def parse_downloaded_filings() -> list[dict]:
    """
    Parse all downloaded full-submission.txt files into text chunks.
    Directory structure: data/raw/sec_filings/sec-edgar-filings/{TICKER}/10-K/{accession}/full-submission.txt
    """
    from src.data_ingestion.chunker import split_text

    base = config.RAW_SEC_DIR / "sec-edgar-filings"
    if not base.exists():
        log.warning("No EDGAR files found at %s — skipping EDGAR parsing", base)
        return []

    chunks = []
    for ticker_dir in sorted(base.iterdir()):
        ticker = ticker_dir.name
        for filing_type_dir in ticker_dir.iterdir():
            for accession_dir in filing_type_dir.iterdir():
                for fpath in accession_dir.glob("*.txt"):
                    texts = _extract_html_from_submission(fpath)
                    for text in texts:
                        for chunk_text in split_text(text):
                            chunks.append({
                                "text": chunk_text,
                                "metadata": {
                                    "domain":       "finance",
                                    "ticker":       ticker,
                                    "source":       f"edgar:{ticker}/{accession_dir.name}",
                                    "filing_type":  "10-K",
                                    "type":         "text",
                                },
                            })

    log.info("Parsed %d chunks from EDGAR full-submission.txt files", len(chunks))
    return chunks


# ── HuggingFace supplement ─────────────────────────────────────────────────────

def load_finance_qa_pairs() -> list[dict]:
    """4K ready-made QA pairs from virattt/financial-qa-10K."""
    log.info("Loading virattt/financial-qa-10K from HuggingFace...")
    ds = load_dataset("virattt/financial-qa-10K", trust_remote_code=True)
    pairs = []
    for split in ds:
        for item in ds[split]:
            q = item.get("question", "")
            a = item.get("answer", "")
            c = item.get("context", "")
            if q and a:
                pairs.append({
                    "question": q, "answer": a, "context": c,
                    "metadata": {"domain": "finance", "source": "financial-qa-10K"},
                })
    log.info("Loaded %d finance QA pairs", len(pairs))
    return pairs


def load_financebench_eval() -> list[dict]:
    """PatronusAI/financebench — gold-standard eval QA with evidence strings."""
    log.info("Loading PatronusAI/financebench from HuggingFace...")
    try:
        ds = load_dataset("PatronusAI/financebench", trust_remote_code=True)
    except Exception as e:
        log.warning("Could not load financebench: %s — using financial-qa-10K for eval instead", e)
        return []

    items = []
    for split in ds:
        for item in ds[split]:
            q = item.get("question", "")
            a = item.get("answer", "")
            evidence = item.get("evidence", [])
            if isinstance(evidence, list):
                ctx = " ".join(e.get("evidence_text", "") for e in evidence if isinstance(e, dict))
            else:
                ctx = str(evidence)
            if q and a:
                items.append({
                    "question":     q,
                    "ground_truth": a,
                    "contexts":     [ctx] if ctx else [],
                    "metadata": {
                        "domain":  "finance",
                        "source":  "financebench",
                        "company": item.get("company_name", ""),
                    },
                })
    log.info("Loaded %d FinanceBench eval items", len(items))
    return items


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_finance_data() -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Parse EDGAR files (already downloaded)
    edgar_chunks = parse_downloaded_filings()

    # 2. Load HuggingFace QA pairs; add their contexts as extra corpus chunks
    qa_pairs = load_finance_qa_pairs()
    from src.data_ingestion.chunker import split_text
    hf_chunks = []
    for qa in qa_pairs:
        if qa["context"]:
            for chunk_text in split_text(qa["context"]):
                hf_chunks.append({
                    "text": chunk_text,
                    "metadata": {"domain": "finance", "source": "financial-qa-10K", "type": "text"},
                })

    all_chunks = edgar_chunks + hf_chunks
    log.info("Total finance chunks: %d (EDGAR=%d, HF=%d)", len(all_chunks), len(edgar_chunks), len(hf_chunks))

    with open(config.FINANCE_CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")
    log.info("Wrote %d finance chunks → %s", len(all_chunks), config.FINANCE_CHUNKS_PATH)

    # 3. Save training QA pairs
    qa_out = config.BASE_DIR / "training" / "data" / "finance_qa_raw.json"
    qa_out.parent.mkdir(parents=True, exist_ok=True)
    with open(qa_out, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2)
    log.info("Saved %d finance QA training pairs → %s", len(qa_pairs), qa_out)

    # 4. Save eval set
    import random
    eval_items = load_financebench_eval()
    if not eval_items:
        # Fallback: build eval from QA pairs
        eval_items = [
            {"question": p["question"], "ground_truth": p["answer"],
             "contexts": [p["context"]] if p["context"] else [],
             "metadata": p["metadata"]}
            for p in qa_pairs if p["question"] and p["answer"]
        ]
    eval_sample = random.sample(eval_items, min(100, len(eval_items)))
    with open(config.EVAL_FINANCE_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_sample, f, indent=2)
    log.info("Saved %d finance eval items → %s", len(eval_sample), config.EVAL_FINANCE_PATH)


if __name__ == "__main__":
    process_finance_data()
