"""Docling runner: convert office/PDF documents to markdown + images."""

from __future__ import annotations

from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc import ImageRefMode


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

