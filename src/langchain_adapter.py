"""Adapters to convert chunks into LangChain documents and load into vector stores."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

from .chunking import ChunkingConfig
from .markdown_pipeline import parse_markdown_and_chunk


def to_langchain_documents(chunks: List[dict]):
    """
    Convert chunk dicts to LangChain Document objects.

    Usage:
        from langchain_core.documents import Document
        documents = to_langchain_documents(chunks)
        vector_store.add_documents(documents, ids=...)
    """
    try:
        from langchain_core.documents import Document
    except ImportError as exc:
        raise ImportError(
            "Install langchain-core to use to_langchain_documents: pip install langchain-core"
        ) from exc

    return [
        Document(page_content=c["page_content"], metadata=c.get("metadata", {}))
        for c in chunks
    ]


def load_file_into_vector_store(
    filepath: Union[str, Path],
    vector_store: Any,
    config: Optional[ChunkingConfig] = None,
    ids: Optional[List[str]] = None,
):
    """
    Chunk an existing markdown file and add to a LangChain vector_store.

    The vector_store must expose `add_documents(documents, ids=...)`
    (e.g. FAISS, Chroma, etc.).
    """
    chunks = parse_markdown_and_chunk(filepath, config=config)
    docs = to_langchain_documents(chunks)
    if ids is None:
        from uuid import uuid4

        ids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(documents=docs, ids=ids)
    return ids

