"""
End-to-end pipeline run for a single document:
- Convert DOCX (via Docling) -> Markdown + images
- Run markdown chunking (DeepDoc-style) using src.core

Usage (from repo root):

    PYTHONPATH=. python data_processing/pipeline_run.py \
        --input data_processing/example_4.1.docx \
        --output-dir data_processing/output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.chunking import ChunkingConfig
from src.core.pipeline import run_docling_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Docling -> markdown -> chunking pipeline.")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data_processing/example_4.1.docx"),
        help="Path to input document (default: data_processing/example_4.1.docx).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_processing/output"),
        help="Directory to store markdown + chunks JSON (default: data_processing/output).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_path: Path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    out_dir: Path = args.output_dir

    cfg = ChunkingConfig(
        chunk_token_num=512,
        overlapped_percent=10.0,
        delimiter="\n",
        custom_delimiters=["\n### ", "\n## ", "### ", "## "],
        table_context_size=50,
        image_context_size=30,
        attach_media_context=True,
        enable_child_chunks=True,
        include_children_in_output=True,
        attach_images_to_section=True,
    )

    json_path = run_docling_pipeline(input_path, out_dir, config=cfg)
    print(f"Pipeline completed. Chunks JSON: {json_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

