"""High-level Docling pipeline: DOCX/PDF -> markdown -> chunks JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from docling.document_converter import DocumentConverter
from docling_core.types.doc import ImageRefMode

from .chunking import ChunkingConfig
from .markdown_pipeline import parse_markdown_and_chunk


def convert_to_markdown_with_docling(input_doc: Path, output_dir: Path) -> Path:
    """
    Use Docling to convert a DOCX/PDF/etc. file to markdown + images.

    Returns:
        Path to the generated markdown file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_converter = DocumentConverter()
    conv_res = doc_converter.convert(input_doc)

    md_path = output_dir / "output.md"
    images_dir = Path("images")  # relative inside output_dir

    conv_res.document.save_as_markdown(
        md_path,
        artifacts_dir=images_dir,
        image_mode=ImageRefMode.REFERENCED,
        image_placeholder="![image](image-not-available)",
    )
    return md_path


def run_docling_pipeline(
    input_path: Path,
    output_dir: Path,
    config: Optional[ChunkingConfig] = None,
) -> Path:
    """
    High-level pipeline:
    1) Docling convert -> markdown
    2) markdown chunking -> chunks JSON

    Returns:
        Path to the written JSON file of chunks.
    """
    cfg = config or ChunkingConfig(
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

    md_path = convert_to_markdown_with_docling(input_path, output_dir)
    chunks = parse_markdown_and_chunk(md_path, config=cfg)

    out_json = output_dir / "pipeline_chunks.json"
    payload = {
        "source_document": str(input_path.resolve()),
        "markdown_path": str(md_path.resolve()),
        "config": cfg.__dict__,
        "num_chunks": len(chunks),
        "chunk_type_sequence": [c["metadata"].get("chunk_type") for c in chunks],
        "chunks": chunks,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_json

