"""
Downloads PubMed abstracts via BioPython Entrez + loads PubMedQA from HuggingFace.
Run: python -m src.data_ingestion.pubmed_downloader
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import logging
import time
from pathlib import Path

from Bio import Entrez, Medline
from datasets import load_dataset

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _fetch_pubmed_ids(query: str, max_results: int) -> list[str]:
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def _fetch_abstracts(pmids: list[str], batch_size: int = 200) -> list[dict]:
    records = []
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        handle = Entrez.efetch(db="pubmed", id=",".join(batch), rettype="medline", retmode="text")
        batch_records = list(Medline.parse(handle))
        handle.close()
        records.extend(batch_records)
        time.sleep(0.34)  # NCBI rate limit: 3 requests/sec
        log.info("  fetched %d / %d abstracts", min(i + batch_size, len(pmids)), len(pmids))
    return records


def download_pubmed() -> list[dict]:
    """Fetch abstracts for all configured queries and return chunk dicts."""
    Entrez.email = config.ENTREZ_EMAIL
    config.RAW_PUBMED_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    seen_pmids: set[str] = set()

    for query in config.PUBMED_QUERIES:
        log.info("Querying PubMed: %s", query)
        pmids = _fetch_pubmed_ids(query, config.PUBMED_MAX_RESULTS)
        new_pmids = [p for p in pmids if p not in seen_pmids]
        seen_pmids.update(new_pmids)
        log.info("  %d new abstracts for query", len(new_pmids))

        records = _fetch_abstracts(new_pmids)
        for rec in records:
            abstract = rec.get("AB", "")
            title    = rec.get("TI", "")
            if not abstract:
                continue
            all_chunks.append({
                "text": f"{title}\n\n{abstract}",
                "metadata": {
                    "domain":   "medical",
                    "source":   f"pubmed:{rec.get('PMID', '')}",
                    "pmid":     rec.get("PMID", ""),
                    "title":    title,
                    "journal":  rec.get("TA", ""),
                    "year":     rec.get("DP", "")[:4] if rec.get("DP") else "",
                    "query":    query,
                    "type":     "abstract",
                },
            })

    # Save raw abstracts
    out_path = config.RAW_PUBMED_DIR / "abstracts.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")
    log.info("Saved %d PubMed abstracts → %s", len(all_chunks), out_path)
    return all_chunks


def load_pubmedqa() -> list[dict]:
    """Load PubMedQA from HuggingFace — free, no auth needed."""
    log.info("Loading PubMedQA dataset from HuggingFace...")
    ds = load_dataset("pubmed_qa", "pqa_labeled", trust_remote_code=True)
    chunks = []
    for split in ds:
        for item in ds[split]:
            context = " ".join(item.get("context", {}).get("contexts", []))
            question = item.get("question", "")
            answer   = item.get("long_answer", "")
            if context:
                chunks.append({
                    "text": context,
                    "metadata": {
                        "domain":   "medical",
                        "source":   "pubmedqa",
                        "question": question,
                        "answer":   answer,
                        "type":     "pubmedqa_context",
                    },
                })
    log.info("Loaded %d PubMedQA contexts", len(chunks))
    return chunks


def process_medical_chunks() -> None:
    """Combine PubMed abstracts + PubMedQA, split text, write JSONL."""
    from src.data_ingestion.chunker import split_text

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pubmed_chunks  = download_pubmed()
    pubmedqa_chunks = load_pubmedqa()
    all_raw = pubmed_chunks + pubmedqa_chunks

    final_chunks = []
    for item in all_raw:
        for chunk_text in split_text(item["text"]):
            final_chunks.append({"text": chunk_text, "metadata": item["metadata"]})

    with open(config.MEDICAL_CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in final_chunks:
            f.write(json.dumps(chunk) + "\n")

    log.info("Wrote %d medical chunks → %s", len(final_chunks), config.MEDICAL_CHUNKS_PATH)


if __name__ == "__main__":
    process_medical_chunks()
