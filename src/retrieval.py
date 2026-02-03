"""
FAISS-based retrieval: ingest chunks -> save local -> load -> search.

Hybrid search (DeepDoc-style): vector (FAISS) + keyword (BM25) + RRF fusion.
Compatible with RAG-style flow: chunk_id, mom_id, chunk_order in metadata for
retrieval_by_children. Embeddings are injectable (e.g. from RAG config or env).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

from loguru import logger

DEFAULT_STORE_PATH = "faiss_index"
CHUNKS_FILENAME = "chunks.json"


def _similarity_range(hits: List[dict]) -> Tuple[float, float]:
    if not hits:
        return 0.0, 0.0
    sims = [h.get("similarity") for h in hits if h.get("similarity") is not None]
    if not sims:
        return 0.0, 0.0
    return min(sims), max(sims)


def _tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize for BM25: words (incl. Vietnamese) and digits."""
    if not text:
        return []
    return re.findall(r"[\wÀ-ỹ]+", text.lower())


def get_embeddings():
    """
    Build embeddings from env (optional). Prefer passing embeddings explicitly
    to retrieval/ingest. Uses OPENAI_API_KEY or API_KEY_EMBEDDING / BASE_URL_EMBEDDING / MODEL_EMBEDDING.
    """
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError(
                "Install langchain-openai or langchain-community with an embedding model; "
                "or pass embeddings=... to retrieval/ingest."
            )
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY_EMBEDDING")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY or API_KEY_EMBEDDING for default embeddings")
    base_url = os.getenv("OPENAI_API_BASE") or os.getenv("BASE_URL_EMBEDDING")
    model = os.getenv("OPENAI_EMBEDDING_MODEL") or os.getenv("MODEL_EMBEDDING") or "text-embedding-3-small"
    kwargs = {"model": model}
    if base_url:
        kwargs["openai_api_base"] = base_url
    return OpenAIEmbeddings(openai_api_key=api_key, **kwargs)


def chunks_to_documents(chunks: List[dict]) -> List[Any]:
    """Convert pipeline chunks to LangChain Documents, preserving chunk_id, mom_id, chunk_order."""
    try:
        from langchain_core.documents import Document
    except ImportError as e:
        raise ImportError("Install langchain-core for Document") from e
    docs = []
    for c in chunks:
        meta = dict(c.get("metadata") or {})
        # Ensure keys needed for retrieval_by_children
        if "chunk_id" not in meta and c.get("metadata", {}).get("chunk_id"):
            meta["chunk_id"] = c["metadata"]["chunk_id"]
        if "mom_id" not in meta:
            meta["mom_id"] = (c.get("metadata") or {}).get("mom_id", "")
        if "chunk_order" not in meta:
            meta["chunk_order"] = (c.get("metadata") or {}).get("chunk_index", 0)
        docs.append(Document(page_content=c.get("page_content", ""), metadata=meta))
    return docs


def _build_bm25_index(chunks: List[dict]) -> Tuple[Any, List[List[str]]]:
    """Build BM25Okapi from chunk page_content. Returns (bm25, tokenized_corpus)."""
    from rank_bm25 import BM25Okapi

    corpus_tokens = [_tokenize_for_bm25(c.get("page_content", "") or "") for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    return bm25, corpus_tokens


def _lexical_search(
    query: str,
    chunks: List[dict],
    bm25: Any,
    top_k: int = 50,
) -> List[dict]:
    """BM25 search over chunks. Returns list of dict: chunk_id, content, metadata, similarity (BM25 score)."""
    if not query or not chunks or bm25 is None:
        return []
    q_tokens = _tokenize_for_bm25(query)
    if not q_tokens:
        return []
    scores = bm25.get_scores(q_tokens)
    indexed = [(i, float(s)) for i, s in enumerate(scores) if s > 0]
    indexed.sort(key=lambda x: x[1], reverse=True)
    hits = []
    for i, score in indexed[:top_k]:
        c = chunks[i]
        meta = dict(c.get("metadata") or {})
        chunk_id = meta.get("chunk_id") or c.get("chunk_id") or str(i)
        hits.append({
            "page_content": c.get("page_content", ""),
            "content": c.get("page_content", ""),
            "metadata": meta,
            "chunk_id": chunk_id,
            "similarity": score,
        })
    return hits


def rrf_fusion(
    dense_hits: List[dict],
    lex_hits: List[dict],
    k_rrf: int = 60,
    k: Optional[int] = None,
) -> List[dict]:
    """
    Reciprocal Rank Fusion: merge dense + lexical by chunk_id.
    score(chunk_id) = sum 1/(k_rrf + rank). Returns list sorted by RRF score desc.
    """
    def key_of(h: dict) -> str:
        cid = h.get("chunk_id") or (h.get("metadata") or {}).get("chunk_id")
        if cid:
            return str(cid)
        content = (h.get("content") or h.get("page_content") or "")[:500]
        return f"_hash_{hash(content)}"

    scores: dict = {}
    registry: dict = {}

    for rank, h in enumerate(dense_hits, start=1):
        key = key_of(h)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k_rrf + rank)
        registry[key] = dict(h)

    for rank, h in enumerate(lex_hits, start=1):
        key = key_of(h)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k_rrf + rank)
        if key not in registry:
            registry[key] = dict(h)

    for key, hit in registry.items():
        hit["similarity"] = scores.get(key, 0.0)

    merged = sorted(registry.values(), key=lambda x: x.get("similarity", 0.0), reverse=True)
    if k is not None:
        merged = merged[:k]
    return merged


def ingest_to_faiss(
    chunks: List[dict],
    embeddings: Any,
    store_path: str | Path = DEFAULT_STORE_PATH,
) -> List[str]:
    """
    Ingest chunks into FAISS and save to local directory.
    Also saves chunks to store_path/chunks.json for BM25 (hybrid search).
    Uses metadata.chunk_id as document id for stable retrieval.
    Returns list of chunk_ids added.
    """
    from langchain_community.vectorstores import FAISS

    store_path = Path(store_path)
    docs = chunks_to_documents(chunks)
    ids = [str((c.get("metadata") or {}).get("chunk_id", i)) for i, c in enumerate(chunks)]
    while len(ids) < len(docs):
        ids.append(str(len(ids)))
    vector_store = FAISS.from_documents(docs, embeddings, ids=ids)
    store_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(store_path))
    # Save chunks for BM25 at retrieval time
    chunks_path = store_path / CHUNKS_FILENAME
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return ids


def load_faiss_store(
    store_path: str | Path = DEFAULT_STORE_PATH,
    embeddings: Optional[Any] = None,
):
    """Load FAISS index from local directory (same API as RAG vector_store)."""
    from langchain_community.vectorstores import FAISS

    store_path = Path(store_path)
    if embeddings is None:
        embeddings = get_embeddings()
    return FAISS.load_local(
        str(store_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_chunks_from_store(store_path: str | Path = DEFAULT_STORE_PATH) -> List[dict]:
    """Load chunks from store_path/chunks.json (saved at ingest). Returns [] if missing."""
    path = Path(store_path) / CHUNKS_FILENAME
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, list) else raw.get("chunks", [])


# DeepDoc-style defaults (RAGFlow retrieval)
PAGERANK_FLD = "pagerank_fea"
DEFAULT_SIMILARITY_THRESHOLD = 0.2
DEFAULT_RANK_FEATURE = {PAGERANK_FLD: 10}


def _apply_rank_feature(hits: List[dict], rank_feature: Optional[dict]) -> None:
    """Add rank_feature boost to similarity (DeepDoc-style). In-place."""
    if not rank_feature or not hits:
        return
    for h in hits:
        meta = h.get("metadata") or {}
        for field, boost in rank_feature.items():
            val = meta.get(field)
            if val is not None and isinstance(val, (int, float)):
                h["similarity"] = (h.get("similarity") or 0.0) + float(val) * float(boost)


def _rerank_hits(
    query: str,
    hits: List[dict],
    reranker: Any,
) -> List[dict]:
    """
    Rerank hits using reranker. Reranker must have .similarity(query, texts)
    returning (scores, ) or just scores (list/array of float).
    """
    if not hits or reranker is None:
        return hits
    texts = [(h.get("content") or h.get("page_content") or "").strip() for h in hits]
    try:
        out = getattr(reranker, "similarity", None)
        if callable(out):
            res = out(query, texts)
            scores = res[0] if isinstance(res, (tuple, list)) and len(res) > 0 else res
        else:
            return hits
    except Exception:
        return hits
    try:
        import numpy as np
        scores = np.asarray(scores).ravel()
    except Exception:
        scores = list(scores) if scores is not None else []
    if len(scores) != len(hits):
        return hits
    for i, h in enumerate(hits):
        h["similarity"] = float(scores[i])
    hits.sort(key=lambda x: x.get("similarity") or 0.0, reverse=True)
    return hits


def retrieval(
    query: str,
    store_path: str | Path = DEFAULT_STORE_PATH,
    embeddings: Optional[Any] = None,
    k: int = 10,
    use_retrieval_by_children: bool = False,
    use_hybrid: bool = True,
    top_k_lex: int = 50,
    k_rrf: int = 60,
    similarity_threshold: float = 0.0,
    reranker: Optional[Any] = None,
    rank_feature: Optional[dict] = None,
) -> List[dict]:
    """
    Hybrid search (DeepDoc-style): vector (FAISS) + keyword (BM25) + RRF fusion,
    optional rank_feature boost, optional reranker, similarity_threshold filter,
    then retrieval_by_children.

    Args:
        similarity_threshold: Min similarity to keep (DeepDoc default 0.2); 0 = no filter.
        reranker: Optional object with .similarity(query, texts) -> (scores,) or scores.
        rank_feature: Optional dict e.g. {"pagerank_fea": 10} to add metadata boost to score.

    Returns list of dicts: content, metadata, similarity.
    """
    store_path = Path(store_path)
    q_short = query[:80] + "..." if len(query) > 80 else query
    logger.info(
        "[retrieval] query={!r}, k={}, use_hybrid={}, use_retrieval_by_children={}, similarity_threshold={}",
        q_short, k, use_hybrid, use_retrieval_by_children, similarity_threshold,
    )

    vs = load_faiss_store(store_path=store_path, embeddings=embeddings)
    logger.info("[retrieval] step=load_faiss_store ok, store_path={}", store_path)

    fetch_k = k * 2 if use_retrieval_by_children else k
    if use_hybrid:
        fetch_k = max(fetch_k, top_k_lex)
    docs_with_scores = vs.similarity_search_with_score(query, k=fetch_k)
    dense_hits = []
    for doc, score in docs_with_scores:
        sim = float(1.0 / (1.0 + abs(score))) if score is not None else 0.0
        dense_hits.append({
            "page_content": doc.page_content,
            "content": doc.page_content,
            "metadata": dict(doc.metadata),
            "chunk_id": doc.metadata.get("chunk_id"),
            "similarity": sim,
        })
    d_min, d_max = _similarity_range(dense_hits)
    logger.info(
        "[retrieval] step=vector_search count={} fetch_k={} similarity_range=[{:.4f}, {:.4f}]",
        len(dense_hits), fetch_k, d_min, d_max,
    )

    chunks = load_chunks_from_store(store_path)
    logger.info("[retrieval] step=load_chunks_from_store count={}", len(chunks))

    if use_hybrid and chunks and query.strip():
        bm25, _ = _build_bm25_index(chunks)
        lex_hits = _lexical_search(query, chunks, bm25, top_k=top_k_lex)
        l_min, l_max = _similarity_range(lex_hits)
        logger.info(
            "[retrieval] step=lexical_search (BM25) count={} top_k_lex={} score_range=[{:.4f}, {:.4f}]",
            len(lex_hits), top_k_lex, l_min, l_max,
        )
        hits = rrf_fusion(dense_hits, lex_hits, k_rrf=k_rrf, k=None)
        r_min, r_max = _similarity_range(hits)
        logger.info(
            "[retrieval] step=rrf_fusion count={} k_rrf={} similarity_range=[{:.4f}, {:.4f}]",
            len(hits), k_rrf, r_min, r_max,
        )
        hits = hits[:k * 2 if use_retrieval_by_children else k]
        logger.info("[retrieval] step=rrf_fusion trim count={}", len(hits))
    else:
        hits = dense_hits[:k * 2 if use_retrieval_by_children else k]
        logger.info("[retrieval] step=hybrid_skipped using dense_hits only count={}", len(hits))

    if rank_feature:
        b_min, b_max = _similarity_range(hits)
        _apply_rank_feature(hits, rank_feature)
        a_min, a_max = _similarity_range(hits)
        logger.info(
            "[retrieval] step=rank_feature applied={} before=[{:.4f}, {:.4f}] after=[{:.4f}, {:.4f}]",
            rank_feature, b_min, b_max, a_min, a_max,
        )
    else:
        logger.info("[retrieval] step=rank_feature skipped (rank_feature=None)")

    if reranker is not None:
        n_before = len(hits)
        hits = _rerank_hits(query, hits, reranker)
        r_min, r_max = _similarity_range(hits)
        logger.info(
            "[retrieval] step=rerank count_before={} count_after={} similarity_range=[{:.4f}, {:.4f}]",
            n_before, len(hits), r_min, r_max,
        )
    else:
        logger.info("[retrieval] step=rerank skipped (reranker=None)")

    if similarity_threshold > 0:
        n_before = len(hits)
        hits = [h for h in hits if (h.get("similarity") or 0.0) >= similarity_threshold]
        logger.info(
            "[retrieval] step=similarity_threshold threshold={:.4f} count_before={} count_after={}",
            similarity_threshold, n_before, len(hits),
        )
    else:
        logger.info("[retrieval] step=similarity_threshold skipped (threshold=0)")

    if use_retrieval_by_children:
        n_before = len(hits)
        from .retrieval_by_children import retrieval_by_children
        hits = retrieval_by_children(hits)
        logger.info("[retrieval] step=retrieval_by_children count_before={} count_after={}", n_before, len(hits))
    else:
        logger.info("[retrieval] step=retrieval_by_children skipped")

    hits = hits[:k]
    logger.info("[retrieval] step=final return count={}", len(hits))
    return hits


def load_chunks_from_pipeline_json(path: str | Path) -> List[dict]:
    """Load chunks array from pipeline_chunks.json (output of run_docling_pipeline)."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("chunks", raw) if isinstance(raw, dict) else raw
