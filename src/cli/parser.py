"""CLI argument parser for data_processing (src-facing)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CliConfig:
    input_md: Path
    output_json: Path | None
    chunk_token_num: int
    overlapped_percent: float
    table_context_size: int
    image_context_size: int
    with_children: bool


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="data_processing",
        description="Markdown-first chunking (DeepDoc-style) for RAG ingestion.",
    )
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to Docling-produced markdown file (e.g. data_processing/output/output.md).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional JSON output path for debug/inspection. If omitted, prints summary only.",
    )
    p.add_argument(
        "--chunk-token-num",
        type=int,
        default=512,
        help="Max tokens per parent text chunk (default: 512).",
    )
    p.add_argument(
        "--overlapped-percent",
        type=float,
        default=0.0,
        help="Overlap percent between parent chunks (0-100, default: 0).",
    )
    p.add_argument(
        "--table-context-size",
        type=int,
        default=50,
        help="Token budget for table context_before/context_after (default: 50).",
    )
    p.add_argument(
        "--image-context-size",
        type=int,
        default=30,
        help="Token budget for image context_before/context_after (default: 30).",
    )
    p.add_argument(
        "--with-children",
        action="store_true",
        help="Enable child chunks (per-sentence/step) in addition to parent text chunks.",
    )
    return p


def parse_cli_args(argv: list[str] | None = None) -> CliConfig:
    parser = build_arg_parser()
    ns = parser.parse_args(argv)
    return CliConfig(
        input_md=ns.input,
        output_json=ns.output,
        chunk_token_num=ns.chunk_token_num,
        overlapped_percent=ns.overlapped_percent,
        table_context_size=ns.table_context_size,
        image_context_size=ns.image_context_size,
        with_children=ns.with_children,
    )


