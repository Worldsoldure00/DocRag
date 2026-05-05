"""
Text chunking utilities — shared by finance and medical pipelines.

Uses a manual recursive character splitter to avoid langchain_text_splitters
which crashes on Python 3.13 due to a dataclasses bug in its Tokenizer class.
"""
from langchain_core.documents import Document
import config


def _recursive_split(text: str, separators: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    sep = separators[0] if separators else ""
    remaining_seps = separators[1:] if separators else []

    if not sep or len(text) <= chunk_size:
        if len(text) <= chunk_size:
            return [text] if text.strip() else []
        if not remaining_seps:
            # hard split
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            return chunks
        return _recursive_split(text, remaining_seps, chunk_size, chunk_overlap)

    splits = text.split(sep)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for split in splits:
        split_len = len(split)
        join_len = len(sep) if current else 0

        if current_len + join_len + split_len > chunk_size and current:
            merged = sep.join(current)
            if merged.strip():
                if len(merged) > chunk_size and remaining_seps:
                    chunks.extend(_recursive_split(merged, remaining_seps, chunk_size, chunk_overlap))
                else:
                    chunks.append(merged)
            # keep overlap
            while current and current_len > chunk_overlap:
                removed = current.pop(0)
                current_len -= len(removed) + len(sep)

        current.append(split)
        current_len += split_len + (len(sep) if len(current) > 1 else 0)

    if current:
        merged = sep.join(current)
        if merged.strip():
            if len(merged) > chunk_size and remaining_seps:
                chunks.extend(_recursive_split(merged, remaining_seps, chunk_size, chunk_overlap))
            else:
                chunks.append(merged)
    return chunks


_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def split_text(text: str) -> list[str]:
    """Split a raw string into overlapping chunks."""
    return _recursive_split(text, _SEPARATORS, config.CHUNK_SIZE, config.CHUNK_OVERLAP)


def split_documents(docs: list[Document]) -> list[Document]:
    """Split a list of LangChain Documents, preserving metadata."""
    result = []
    for doc in docs:
        for chunk in split_text(doc.page_content):
            result.append(Document(page_content=chunk, metadata=doc.metadata.copy()))
    return result


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
