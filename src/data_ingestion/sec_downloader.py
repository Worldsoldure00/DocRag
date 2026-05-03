"""
Downloads SEC 10-K filings via sec-edgar-downloader and converts them to text chunks.
Run: python -m src.data_ingestion.sec_downloader
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import logging
from pathlib import Path

import pdfplumber
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BOILERPLATE_PHRASES = [
    "table of contents",
    "forward-looking statements",
    "this annual report on form 10-k",
    "incorporated by reference",
    "exhibit index",
]


def _strip_boilerplate(text: str) -> str:
    lines = text.splitlines()
    kept = []
    for line in lines:
        lower = line.lower().strip()
        if any(phrase in lower for phrase in BOILERPLATE_PHRASES):
            continue
        if len(lower) < 4:
            continue
        kept.append(line)
    return "\n".join(kept)


def _parse_html_filing(filepath: Path) -> str:
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _parse_pdf_filing(filepath: Path) -> tuple[str, list[str]]:
    """Returns (plain_text, list_of_table_markdown_strings)."""
    text_parts = []
    tables = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
            for table in page.extract_tables():
                if table:
                    header = " | ".join(str(c) for c in (table[0] or []))
                    rows = [" | ".join(str(c or "") for c in row) for row in table[1:]]
                    tables.append(header + "\n" + "\n".join(rows))
    return "\n".join(text_parts), tables


def download_filings() -> None:
    """Download 10-K filings for all configured tickers."""
    name  = config.SEC_EDGAR_NAME  or "DocSight RAG"
    email = config.SEC_EDGAR_EMAIL or "placeholder@email.com"

    if "placeholder" in email:
        log.warning("SEC_EDGAR_NAME / SEC_EDGAR_EMAIL not set in .env — using placeholders.")

    dl = Downloader(name, email, download_folder=str(config.RAW_SEC_DIR))
    for ticker in config.SEC_TICKERS:
        log.info("Downloading %s %s", ticker, config.SEC_FILING_TYPE)
        dl.get(
            config.SEC_FILING_TYPE,
            ticker,
            after=config.SEC_DATE_AFTER,
            before=config.SEC_DATE_BEFORE,
        )
    log.info("Downloads complete → %s", config.RAW_SEC_DIR)


def process_filings() -> list[dict]:
    """Parse downloaded filings into chunk dicts and write to JSONL."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    chunks = []

    for ticker_dir in sorted(config.RAW_SEC_DIR.glob("*")):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        for filing_dir in sorted(ticker_dir.glob("**")):
            for fpath in filing_dir.glob("*"):
                if fpath.suffix in (".htm", ".html"):
                    raw_text = _parse_html_filing(fpath)
                    table_chunks = []
                elif fpath.suffix == ".pdf":
                    raw_text, table_chunks = _parse_pdf_filing(fpath)
                else:
                    continue

                clean = _strip_boilerplate(raw_text)
                base_meta = {
                    "domain": "finance",
                    "ticker": ticker,
                    "source": str(fpath.relative_to(config.BASE_DIR)),
                    "filing_type": config.SEC_FILING_TYPE,
                }

                # Add table chunks as individual entries (keep each table intact)
                for i, tbl in enumerate(table_chunks):
                    chunks.append({
                        "text": tbl,
                        "metadata": {**base_meta, "section": f"table_{i}", "type": "table"},
                    })

                # Split narrative text
                from src.data_ingestion.chunker import split_text
                for chunk_text in split_text(clean):
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {**base_meta, "type": "text"},
                    })

    with open(config.FINANCE_CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    log.info("Wrote %d finance chunks → %s", len(chunks), config.FINANCE_CHUNKS_PATH)
    return chunks


if __name__ == "__main__":
    download_filings()
    process_filings()
