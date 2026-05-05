"""Bisect the langchain_text_splitters crash."""
import sys
print("step 1: std imports"); sys.stdout.flush()
import copy, logging, abc, dataclasses, enum
from typing import TYPE_CHECKING, Any, Literal, TypeVar
print("step 2: langchain_core"); sys.stdout.flush()
from langchain_core.documents import BaseDocumentTransformer, Document
print("step 3: typing_extensions"); sys.stdout.flush()
from typing_extensions import Self, override
print("step 4: tiktoken"); sys.stdout.flush()
try:
    import tiktoken
    print("  tiktoken ok")
except ImportError:
    print("  tiktoken missing")
sys.stdout.flush()
print("step 5: transformers"); sys.stdout.flush()
try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    print("  transformers ok")
except ImportError:
    print("  transformers missing")
sys.stdout.flush()

print("step 6: define TextSplitter class"); sys.stdout.flush()

class TextSplitter(BaseDocumentTransformer, abc.ABC):
    def __init__(self, chunk_size=4000, chunk_overlap=200):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @abc.abstractmethod
    def split_text(self, text: str): ...

    @override
    def transform_documents(self, documents, **kwargs):
        return list(documents)

print("step 7: TextSplitter defined ok"); sys.stdout.flush()

print("step 8: define Tokenizer dataclass"); sys.stdout.flush()
from dataclasses import dataclass
from collections.abc import Callable

@dataclass(frozen=True)
class Tokenizer:
    chunk_overlap: int
    tokens_per_chunk: int
    decode: Callable[[list[int]], str]
    encode: Callable[[str], list[int]]

print("step 9: Tokenizer defined ok"); sys.stdout.flush()
print("ALL OK"); sys.stdout.flush()
