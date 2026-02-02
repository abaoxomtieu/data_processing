"""Core markdown parser implementation for data_processing."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

from ..md_table_detect import find_tables_in_markdown

_log = logging.getLogger(__name__)

_MD_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
_HTML_IMG_PATTERN = re.compile(r"<img[^>]*?src=[\"']([^\"']+)[\"'][^>]*?>", re.IGNORECASE)


def parse_markdown_blocks(
    md_path: Union[str, Path],
    callback: Optional[Callable[[float, str], None]] = None,
) -> List[dict]:
    """
    Parse a Docling-produced markdown file into ordered blocks preserving markdown order.

    Block schema:
    - text:  {"type":"text","text":str,"start_line":int,"end_line":int,"section_path":[...]}
    - table: {"type":"table","text":str,"start_line":int,"end_line":int,"section_path":[...]}
    - image: {"type":"image","path":str,"caption":str,"line":int,"section_path":[...]}
    """
    path = Path(md_path)
    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {path}")
    if callback:
        callback(0.1, "[MD] Reading markdown file...")
    full_md = path.read_text(encoding="utf-8", errors="replace")
    blocks = _split_markdown_to_blocks(full_md, base_dir=path.parent)
    if callback:
        callback(1.0, f"[MD] Parsed markdown blocks: {len(blocks)}")
    return blocks


def _blocks_to_sections_tables(
    blocks: List[dict],
) -> Tuple[List[Tuple[str, str]], List[Tuple[Tuple[Optional[Any], str], List]]]:
    """Convert ordered blocks to legacy (sections, tables) outputs."""
    sections: List[Tuple[str, str]] = []
    tables: List[Tuple[Tuple[Optional[Any], str], List]] = []
    for b in blocks:
        btype = b.get("type")
        if btype == "text":
            txt = (b.get("text") or "").strip()
            if txt:
                sections.append((txt, ""))  # position_tag currently unused for markdown
        elif btype == "table":
            tables.append(((None, b.get("text") or ""), []))
        elif btype == "image":
            tables.append((({"type": "image", "path": b.get("path", "")}, b.get("caption", "") or ""), []))
    return sections, tables


def _split_markdown_to_blocks(
    full_md: str,
    base_dir: Optional[Path] = None,
) -> List[dict]:
    """
    Split a markdown document into ordered blocks, preserving the original order:
    - {"type": "text", "text": "..."}
    - {"type": "table", "text": "..."}    (GFM pipe tables)
    - {"type": "image", "path": "...", "caption": "..."}  (markdown image refs)

    This is used both for Docling-generated markdown and for pre-existing markdown
    files that have already been produced by Docling. Image/table blocks are
    extracted as standalone blocks, and text blocks contain the remaining text.
    """
    blocks: List[dict] = []

    table_blocks = find_tables_in_markdown(full_md)
    # Map table start line idx (0-based) -> (end_idx_exclusive, raw_lines)
    table_by_start: dict[int, tuple[int, List[str]]] = {}
    for start, end, block in table_blocks:
        table_by_start[start - 1] = (end, block)  # end is 1-based exclusive in find_tables_in_markdown loop logic

    # Image paths in markdown are relative to markdown file location when using save_as_markdown.
    # We convert them to absolute paths to keep downstream stable.

    def _resolve_img(p: str) -> str:
        if base_dir is None:
            return p
        # keep external URLs as-is
        if re.match(r"^https?://", p, flags=re.I):
            return p
        return str((base_dir / p).resolve())

    lines = full_md.splitlines()
    current_text: List[str] = []
    current_start_line = 1
    section_stack: List[Tuple[int, str]] = []  # (level, title)
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    def _flush_text(end_line: int) -> None:
        nonlocal current_text, blocks, current_start_line
        txt = "\n".join(current_text).strip()
        if txt:
            blocks.append(
                {
                    "type": "text",
                    "text": txt,
                    "start_line": current_start_line,
                    "end_line": end_line,
                    "section_path": [t for _, t in section_stack],
                }
            )
        current_text = []
        current_start_line = end_line + 1

    i = 0
    while i < len(lines):
        line_no = i + 1
        # Table block starts here?
        if i in table_by_start:
            _flush_text(line_no - 1)
            end_1based, block = table_by_start[i]
            table_text = "\n".join(block).strip()
            if table_text:
                blocks.append(
                    {
                        "type": "table",
                        "text": table_text,
                        "start_line": line_no,
                        "end_line": end_1based,
                        "section_path": [t for _, t in section_stack],
                    }
                )
            # Skip table lines
            i = end_1based
            current_start_line = end_1based + 1
            continue

        line = lines[i]

        # Track markdown headings as section boundaries (for hierarchical merge + image association).
        hm = heading_re.match(line)
        if hm:
            _flush_text(line_no - 1)
            level = len(hm.group(1))
            title = hm.group(0).strip()
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, title))
            blocks.append(
                {
                    "type": "text",
                    "text": title,
                    "start_line": line_no,
                    "end_line": line_no,
                    "section_path": [t for _, t in section_stack],
                    "is_heading": True,
                    "heading_level": level,
                }
            )
            current_start_line = line_no + 1
            i += 1
            continue

        # Extract one or more images from this line, preserving order.
        m = _MD_IMAGE_PATTERN.search(line)
        h = _HTML_IMG_PATTERN.search(line)
        if not m and not h:
            current_text.append(line)
            i += 1
            continue

        # If image is inline, keep surrounding text as text blocks.
        while True:
            # choose earliest match among md and html img
            m = _MD_IMAGE_PATTERN.search(line)
            h = _HTML_IMG_PATTERN.search(line)
            cand = []
            if m:
                cand.append(("md", m.start(), m))
            if h:
                cand.append(("html", h.start(), h))
            if not cand:
                break
            cand.sort(key=lambda x: x[1])
            kind, _, match = cand[0]

            before = line[: match.start()]
            if before.strip():
                current_text.append(before)
                _flush_text(line_no)
            else:
                _flush_text(line_no)

            if kind == "md":
                caption = (match.group(1) or "").strip()
                img_ref = (match.group(2) or "").strip()
            else:
                caption = ""
                img_ref = (match.group(1) or "").strip()

            if img_ref and img_ref != "image-not-available":
                blocks.append(
                    {
                        "type": "image",
                        "path": _resolve_img(img_ref),
                        "caption": caption,
                        "line": line_no,
                        "section_path": [t for _, t in section_stack],
                    }
                )

            line = line[match.end() :]

        if line.strip():
            current_text.append(line)

        i += 1

    _flush_text(len(lines))
    return blocks


def parse_markdown_file_to_legacy(
    md_path: Union[str, Path],
    callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[List[Tuple[str, str]], List[Tuple[Tuple[Optional[Any], str], List]]]:
    """Legacy adapter (sections, tables) for older callers."""
    blocks = parse_markdown_blocks(md_path, callback=callback)
    return _blocks_to_sections_tables(blocks)


