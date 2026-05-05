"""Standalone FAISS index builder — streams docs in batches to avoid memory spikes."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
# Reduce CUDA memory fragmentation on RTX 4070 (8 GB VRAM)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# sentence_transformers MUST come before torch and faiss (Windows DLL init order)
from sentence_transformers import SentenceTransformer
import torch

print(f"torch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

import faiss
import numpy as np
print(f"faiss {faiss.__version__} ready")

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import config
import json, logging, random
from pathlib import Path
from collections import defaultdict
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_jsonl(path) -> List[Document]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                docs.append(Document(page_content=item["text"], metadata=item["metadata"]))
    return docs


def subsample(docs, max_n):
    if max_n is None or len(docs) <= max_n:
        return docs
    groups = defaultdict(list)
    for d in docs:
        key = d.metadata.get("ticker") or d.metadata.get("source", "unknown")
        groups[key].append(d)
    sampled = []
    per_group = max(1, max_n // len(groups))
    for g in groups.values():
        sampled.extend(random.sample(g, min(per_group, len(g))))
    sampled_set = {id(d) for d in sampled}
    remaining = [d for d in docs if id(d) not in sampled_set]
    if len(sampled) < max_n:
        sampled.extend(random.sample(remaining, min(max_n - len(sampled), len(remaining))))
    return sampled[:max_n]


def build_index_batched(
    model_name: str,
    docs: List[Document],
    device: str,
    out_path: Path,
    encode_batch: int = 512,
    add_batch: int = 5000,
) -> None:
    """
    Build a FAISS index by encoding and adding docs in small batches.
    Avoids passing 60K texts to Python at once (memory spike + crash).
    """
    log.info("Loading model %s on %s ...", model_name, device)
    model = SentenceTransformer(model_name, device=device)
    dim = model.get_embedding_dimension()
    log.info("Model dim=%d", dim)

    # Flat L2 index (we normalize embeddings so cosine ≈ L2)
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine
    all_docs: List[Document] = []

    n = len(docs)
    for start in range(0, n, add_batch):
        batch_docs = docs[start:start + add_batch]
        texts = [d.page_content for d in batch_docs]
        log.info("  Encoding batch %d–%d / %d ...", start, start + len(texts), n)
        vecs = model.encode(
            texts,
            batch_size=encode_batch,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vecs = np.array(vecs, dtype=np.float32)
        index.add(vecs)
        all_docs.extend(batch_docs)
        del vecs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log.info("  Added %d vectors to FAISS index", index.ntotal)

    # Wrap in LangChain FAISS so it can be loaded normally later
    # Build the docstore and index_to_docstore_id that LangChain expects
    from langchain_community.vectorstores.faiss import dependable_faiss_import
    from langchain_community.docstore.in_memory import InMemoryDocstore
    import uuid

    index_to_id = {i: str(uuid.uuid4()) for i in range(len(all_docs))}
    docstore_dict = {index_to_id[i]: all_docs[i] for i in range(len(all_docs))}
    docstore = InMemoryDocstore(docstore_dict)

    # Wrap model as a proper Embeddings object for LangChain FAISS interface
    from langchain_core.embeddings import Embeddings as _BaseEmb

    class _DummyEmb(_BaseEmb):
        def embed_query(self, text: str) -> List[float]:
            v = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
            return v[0].tolist()
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            v = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            return v.tolist()

    lc_faiss = FAISS(
        embedding_function=_DummyEmb(),
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_id,
    )

    out_path.mkdir(parents=True, exist_ok=True)
    lc_faiss.save_local(str(out_path))
    log.info("Index saved → %s  (%d vectors)", out_path, index.ntotal)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Finance index ──────────────────────────────────────────────────────────────
log.info("=== Building FINANCE index ===")
finance_docs = load_jsonl(config.FINANCE_CHUNKS_PATH)
log.info("Loaded %d finance chunks", len(finance_docs))
finance_docs = subsample(finance_docs, config.MAX_FINANCE_CHUNKS)
log.info("Using %d chunks", len(finance_docs))

build_index_batched(
    model_name=config.FINANCE_EMBED_MODEL,
    docs=finance_docs,
    device=DEVICE,
    out_path=config.FINANCE_INDEX_PATH,
    encode_batch=16,
    add_batch=5000,
)
del finance_docs

# ── Medical index ──────────────────────────────────────────────────────────────
log.info("=== Building MEDICAL index ===")
medical_docs = load_jsonl(config.MEDICAL_CHUNKS_PATH)
log.info("Loaded %d medical chunks", len(medical_docs))
medical_docs = subsample(medical_docs, config.MAX_MEDICAL_CHUNKS)
log.info("Using %d chunks", len(medical_docs))

build_index_batched(
    model_name=config.MEDICAL_EMBED_MODEL,
    docs=medical_docs,
    device=DEVICE,
    out_path=config.MEDICAL_INDEX_PATH,
    encode_batch=16,
    add_batch=5000,
)

log.info("=== All indexes built successfully ===")
