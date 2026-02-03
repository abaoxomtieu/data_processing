"""
Run multiple retrieval test cases with loguru logger for debugging.

Usage (from repo root):
  PYTHONPATH=. python data_processing/test_retrieval_cases.py
  PYTHONPATH=. python data_processing/test_retrieval_cases.py --chunks data_processing/output/pipeline_chunks.json --store data_processing/faiss_index
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

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

# Configure loguru: level INFO, format with time
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>: {message}",
    level="INFO",
)

from src.retrieval import (
    get_embeddings,
    ingest_to_faiss,
    load_chunks_from_pipeline_json,
    retrieval,
)

# Test cases: (name, query, k, use_hybrid, use_retrieval_by_children, similarity_threshold, rank_feature)
TEST_CASES = [
    {
        "name": "case_1_hybrid_by_children",
        "query": "cấu hình phân phát scheduler",
        "k": 5,
        "use_hybrid": True,
        "use_retrieval_by_children": True,
        "similarity_threshold": 0.0,
        "rank_feature": None,
    },
    {
        "name": "case_2_vector_only",
        "query": "QoS gán port",
        "k": 5,
        "use_hybrid": False,
        "use_retrieval_by_children": False,
        "similarity_threshold": 0.0,
        "rank_feature": None,
    },
    {
        "name": "case_3_hybrid_no_children",
        "query": "lưu lượng chính sách mạng Metro",
        "k": 5,
        "use_hybrid": True,
        "use_retrieval_by_children": False,
        "similarity_threshold": 0.0,
        "rank_feature": None,
    },
    {
        "name": "case_4_similarity_threshold",
        "query": "cấu hình DSCP",
        "k": 10,
        "use_hybrid": True,
        "use_retrieval_by_children": False,
        "similarity_threshold": 0.01,
        "rank_feature": None,
    },
    {
        "name": "case_5_rank_feature",
        "query": "qui định chung",
        "k": 3,
        "use_hybrid": True,
        "use_retrieval_by_children": False,
        "similarity_threshold": 0.0,
        "rank_feature": {"pagerank_fea": 10},
    },
    {
        "name": "case_6_short_query",
        "query": "DSCP",
        "k": 3,
        "use_hybrid": True,
        "use_retrieval_by_children": True,
        "similarity_threshold": 0.0,
        "rank_feature": None,
    },
    {
        "name": "case_7_empty_like_query",
        "query": "xyz không tồn tại",
        "k": 5,
        "use_hybrid": True,
        "use_retrieval_by_children": False,
        "similarity_threshold": 0.0,
        "rank_feature": None,
    },
]


def run_one(embeddings, store_path: Path, case: dict) -> list:
    hits = retrieval(
        case["query"],
        store_path=store_path,
        embeddings=embeddings,
        k=case["k"],
        use_retrieval_by_children=case["use_retrieval_by_children"],
        use_hybrid=case["use_hybrid"],
        similarity_threshold=case["similarity_threshold"],
        rank_feature=case.get("rank_feature"),
    )
    return hits


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run retrieval test cases with loguru")
    parser.add_argument("--chunks", type=Path, default=Path("data_processing/output/pipeline_chunks.json"))
    parser.add_argument("--store", type=Path, default=Path("data_processing/faiss_index"))
    args = parser.parse_args()

    chunks_path = args.chunks
    store_path = args.store

    if not chunks_path.exists():
        logger.error("Chunks file not found: {}", chunks_path)
        sys.exit(1)

    chunks = load_chunks_from_pipeline_json(chunks_path)
    if not chunks:
        logger.error("No chunks in file")
        sys.exit(1)

    logger.info("===== Ingest: {} chunks into {}", len(chunks), store_path)
    embeddings = get_embeddings()
    ingest_to_faiss(chunks, embeddings, store_path=store_path)
    logger.info("===== Ingest done")

    for case in TEST_CASES:
        name = case["name"]
        logger.info("")
        logger.info("===== Test case: {} =====", name)
        logger.info(
            "  query={!r} k={} use_hybrid={} use_retrieval_by_children={} similarity_threshold={} rank_feature={}",
            case["query"], case["k"], case["use_hybrid"], case["use_retrieval_by_children"],
            case["similarity_threshold"], case.get("rank_feature"),
        )
        try:
            hits = run_one(embeddings, store_path, case)
            logger.info("  result: count={}", len(hits))
            for i, h in enumerate(hits[:3], 1):
                sim = h.get("similarity")
                cid = h.get("chunk_id") or (h.get("metadata") or {}).get("chunk_id")
                content = (h.get("content") or h.get("page_content") or "")[:80]
                logger.info("    hit {}: sim={:.4f} chunk_id={} content={}...", i, sim or 0.0, cid, content)
            if len(hits) > 3:
                logger.info("    ... and {} more", len(hits) - 3)
        except Exception as e:
            logger.exception("  FAILED: {}", e)

    logger.info("")
    logger.info("===== All test cases finished =====")


if __name__ == "__main__":
    main()
