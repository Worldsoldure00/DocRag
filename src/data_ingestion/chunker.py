"""
Text chunking utilities — shared by finance and medical pipelines.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import config


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def split_text(text: str) -> list[str]:
    """Split a raw string into overlapping chunks."""
    return _splitter.split_text(text)


def split_documents(docs: list[Document]) -> list[Document]:
    """Split a list of LangChain Documents, preserving metadata."""
    return _splitter.split_documents(docs)


def jsonl_to_documents(jsonl_path) -> list[Document]:
    """Load a processed JSONL chunks file into LangChain Documents."""
    import json
    from pathlib import Path

    docs = []
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            docs.append(Document(page_content=item["text"], metadata=item["metadata"]))
    return docs
