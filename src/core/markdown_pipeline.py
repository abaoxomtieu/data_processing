"""Markdown-first chunking pipeline: markdown file -> chunks list."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Union

from .chunking import ChunkingConfig, run_chunking_blocks
from .docling_parser import parse_markdown_blocks

_log = logging.getLogger(__name__)


def parse_markdown_and_chunk(
    md_path: Union[str, Path],
    config: Optional[ChunkingConfig] = None,
    callback: Optional[Callable[[float, str], None]] = None,
) -> List[dict]:
    """
    Chunking entrypoint for markdown inputs (typically Docling-parsed .md files).

    This bypasses DocumentConverter and works directly on an existing markdown
    file, but still uses the same DeepDoc-inspired chunking solution:
    - token-based text chunks with configurable chunk_token_num / overlap
    - delimiter/custom delimiters
    - table/image chunks separated with their own metadata
    - configurable table_context_size / image_context_size
    - attach_media_context flag in metadata for downstream retrieval logic.
    """
    config = config or ChunkingConfig()
    md_path = Path(md_path)
    source_name = md_path.name

    blocks = parse_markdown_blocks(md_path, callback=callback)
    chunks = run_chunking_blocks(blocks, config, source_name=source_name)
    _log.info("Markdown pipeline produced %d chunks from %s", len(chunks), source_name)
    return chunks

