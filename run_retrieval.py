"""
Run retrieval test: load pipeline_chunks.json -> ingest to FAISS -> save local -> load -> search.

Usage (from repo root or data_processing):
  PYTHONPATH=. python data_processing/run_retrieval.py \\
    --chunks data_processing/output/pipeline_chunks.json \\
    --store data_processing/faiss_index \\
    --query "Nội dung cần tìm"

Optional: set OPENAI_API_KEY or API_KEY_EMBEDDING (and BASE_URL_EMBEDDING, MODEL_EMBEDDING)
or use HuggingFace (install sentence-transformers) with no key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

# Ensure data_processing can be imported and load .env
_root = Path(__file__).resolve().parent
_env = _root / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env)
    except ImportError:
        pass
if str(_root.parent) not in sys.path:
    sys.path.insert(0, str(_root.parent))
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.retrieval import (
    DEFAULT_RANK_FEATURE,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_STORE_PATH,
    PAGERANK_FLD,
    get_embeddings,
    ingest_to_faiss,
    load_chunks_from_pipeline_json,
    retrieval,
)


def main():
    parser = argparse.ArgumentParser(description="Ingest chunks to FAISS and run retrieval test")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data_processing/output/pipeline_chunks.json"),
        help="Path to pipeline_chunks.json (or chunks.json with 'chunks' key)",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=Path("data_processing/faiss_index"),
        help="Directory to save/load FAISS index (default: data_processing/faiss_index)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Query string to test retrieval (if empty, only ingest)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "--by-children",
        action="store_true",
        help="Group hits by parent (retrieval_by_children) before returning",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only run ingest (save FAISS), do not run retrieval",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Disable hybrid (vector-only, no BM25 + RRF)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.0,
        help=f"Min similarity to keep (0=no filter; DeepDoc uses {DEFAULT_SIMILARITY_THRESHOLD} with normalized scores)",
    )
    parser.add_argument(
        "--rank-feature",
        type=str,
        default="",
        help=f"Rank feature boost e.g. '{PAGERANK_FLD}:10' (metadata field: boost); empty = no boost",
    )
    args = parser.parse_args()

    chunks_path = args.chunks
    if not chunks_path.exists():
        logger.error("Chunks file not found: {}", chunks_path)
        logger.info("Run pipeline first, e.g.: python data_processing/main.py --input ... --output-dir data_processing/output")
        sys.exit(1)

    chunks = load_chunks_from_pipeline_json(chunks_path)
    if not chunks:
        logger.error("No chunks in file")
        sys.exit(1)
    logger.info("Loaded {} chunks from {}", len(chunks), chunks_path)

    embeddings = get_embeddings()
    store_path = args.store
    ids = ingest_to_faiss(chunks, embeddings, store_path=store_path)
    logger.info("Ingested {} chunks into FAISS at {}", len(ids), store_path)

    if args.ingest_only:
        return

    query = args.query
    if not query:
        query = "nội dung tài liệu"
        logger.info("No --query given, using default: {!r}", query)

    rank_feature = None
    if args.rank_feature:
        parts = args.rank_feature.split(":", 1)
        if len(parts) == 2:
            try:
                rank_feature = {parts[0].strip(): float(parts[1].strip())}
            except ValueError:
                rank_feature = DEFAULT_RANK_FEATURE
        else:
            rank_feature = DEFAULT_RANK_FEATURE

    hits = retrieval(
        query,
        store_path=store_path,
        embeddings=embeddings,
        k=args.k,
        use_retrieval_by_children=args.by_children,
        use_hybrid=not args.no_hybrid,
        similarity_threshold=args.similarity_threshold,
        rank_feature=rank_feature,
    )
    logger.info(
        "Retrieval (k={}, hybrid={}, by_children={}, similarity_threshold={})",
        args.k, not args.no_hybrid, args.by_children, args.similarity_threshold,
    )
    for i, h in enumerate(hits, 1):
        sim = h.get("similarity")
        content = (h.get("content") or h.get("page_content") or "")[:200]
        cid = h.get("chunk_id") or h.get("metadata", {}).get("chunk_id")
        logger.info("  {} [sim={:.4f}] chunk_id={}", i, sim or 0.0, cid)
        logger.info("     {}...", content)
    logger.info("Done. Store: {}", store_path.resolve())


if __name__ == "__main__":
    main()
