"""
Post-retrieval: group hits by parent (mom_id) and merge content (RAGFlow-style retrieval_by_children).

Use after hybrid/vector search: pass the list of chunk hits, get back parent-level chunks
with merged content from siblings. Compatible with chunk output that has chunk_id, mom_id
(and optional parent_id) in metadata.

Indexing (for RAG): when writing pipeline_chunks.json chunks to vector/lexical store,
set parent_id = metadata.mom_id if metadata.mom_id else metadata.chunk_id so that
_merge_neighbors or retrieval_by_children can group by parent.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, List


def _content_of(c: dict) -> str:
    return (c.get("content") or c.get("page_content") or "").strip()


def _parent_id_of(c: dict) -> str:
    """Parent id for grouping: mom_id (child → parent) or chunk_id (parent = self)."""
    meta = c.get("metadata") or {}
    pid = meta.get("mom_id") or meta.get("parent_id") or ""
    if pid:
        return pid
    return meta.get("chunk_id") or c.get("chunk_id") or ""


def _chunk_id_of(c: dict) -> str:
    meta = c.get("metadata") or {}
    return meta.get("chunk_id") or c.get("chunk_id") or ""


def _chunk_order_of(c: dict) -> int:
    meta = c.get("metadata") or {}
    return meta.get("chunk_order", meta.get("chunk_index", 0))


def retrieval_by_children(chunks: List[dict]) -> List[dict]:
    """
    Group chunk hits by parent (mom_id / parent_id) and merge content per parent.

    - Chunks with the same mom_id (children of one parent) are merged into one result.
    - Chunks with no mom_id (parents or standalone) stay as-is.
    - Each returned item has: chunk_id (parent id), content or page_content (merged),
      metadata with parent_id = chunk_id, and mean similarity if present.

    Use this after vector/lexical search so that retrieval is "by child" but
    results are returned at parent level with merged sibling content.
    """
    if not chunks:
        return []

    by_parent: dict[Any, List[dict]] = defaultdict(list)
    for c in chunks:
        pid = _parent_id_of(c)
        by_parent[pid].append(c)

    out: List[dict] = []
    for parent_id, group in by_parent.items():
        group_sorted = sorted(group, key=_chunk_order_of)
        contents = [_content_of(c) for c in group_sorted if _content_of(c)]
        merged_content = "\n".join(contents)

        # Build one representative chunk (parent-level)
        first = group_sorted[0]
        meta = dict(first.get("metadata") or {})
        meta["parent_id"] = parent_id
        meta["chunk_id"] = parent_id
        meta["chunk_order"] = _chunk_order_of(first)

        sims = [
            first.get("similarity"),
            *[c.get("similarity") for c in group_sorted[1:] if c.get("similarity") is not None],
        ]
        sims = [s for s in sims if s is not None]
        similarity = sum(sims) / len(sims) if sims else None

        d: dict[str, Any] = {
            "chunk_id": parent_id,
            "parent_id": parent_id,
            "content": merged_content,
            "page_content": merged_content,
            "metadata": meta,
        }
        if similarity is not None:
            d["similarity"] = similarity
        # Preserve common fields from first hit
        for key in ("doc_id", "docnm_kwd", "kb_id", "section_path"):
            if key in first and first[key] is not None:
                d[key] = first[key]
        out.append(d)

    # Sort by similarity descending if present
    if out and out[0].get("similarity") is not None:
        out.sort(key=lambda x: (x.get("similarity") or 0.0), reverse=True)
    return out
