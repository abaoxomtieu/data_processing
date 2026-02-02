"""Core chunking logic (markdown-first, DeepDoc-inspired)."""

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODER = None


def num_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENCODER is None:
        return len(text) // 4
    try:
        return len(_ENCODER.encode(text))
    except Exception:
        return len(text) // 4


# Remove position tags (e.g. @@1\t0.0\t100.0##) before token count
_TAG_PATTERN = re.compile(r"@@[\t0-9.-]+?##")


def remove_position_tag(txt: str) -> str:
    return _TAG_PATTERN.sub("", txt).strip()


def _take_first_tokens(text: str, token_budget: int) -> str:
    text = remove_position_tag(text)
    if token_budget <= 0 or not text:
        return ""
    if _ENCODER is None:
        # rough fallback
        return text[: max(0, token_budget * 4)]
    try:
        ids = _ENCODER.encode(text)
        return _ENCODER.decode(ids[:token_budget])
    except Exception:
        return text[: max(0, token_budget * 4)]


def _take_last_tokens(text: str, token_budget: int) -> str:
    text = remove_position_tag(text)
    if token_budget <= 0 or not text:
        return ""
    if _ENCODER is None:
        # rough fallback
        return text[max(0, len(text) - token_budget * 4) :]
    try:
        ids = _ENCODER.encode(text)
        return _ENCODER.decode(ids[-token_budget:])
    except Exception:
        return text[max(0, len(text) - token_budget * 4) :]


@dataclass
class ChunkingConfig:
    chunk_token_num: int = 512
    delimiter: str = "\n。；！？"
    overlapped_percent: float = 0.0
    table_context_size: int = 0
    # Additional custom delimiters (high precedence, e.g. "### " for headings).
    custom_delimiters: List[str] = field(default_factory=list)
    # Minimum tokens for a chunk to be kept (tiny noise is dropped).
    min_chunk_tokens: int = 8
    # Image-specific context window (tokens) around image/media chunks.
    image_context_size: int = 0
    # Whether media chunks (table/image) are considered to carry attached context.
    # This flag is written into metadata; retrieval layer can choose to treat these
    # chunks specially (e.g. always fetch them together with nearby text chunks).
    attach_media_context: bool = False
    # Parent-child chunking (child chunks used for retrieval)
    # Disabled by default for simpler JSON output; can be turned on explicitly.
    enable_child_chunks: bool = False
    # NOTE: exclude '.' and ':' to avoid splitting numeric patterns like '2.2'
    # or 'Bước 1:' inside headings/bullets. Sentence breaks are mostly on
    # newlines and strong punctuation for our markdown data.
    children_delimiters_pattern: str = r"([。！？!?；;\n])"
    include_children_in_output: bool = False
    # Hierarchical merge (markdown outline)
    hierarchical_merge: bool = False
    hierarchical_depth: int = 3
    # Markdown: attach images to nearest preceding section/text chunk
    attach_images_to_section: bool = True


def _split_sentences(text: str, pattern: str) -> List[str]:
    parts = re.split(pattern, text)
    sents: List[str] = []
    buf = ""
    for p in parts:
        if not p:
            continue
        if re.fullmatch(pattern, p):
            buf += p
            sents.append(buf)
            buf = ""
        else:
            buf += p
    if buf:
        sents.append(buf)
    return sents


def _trim_to_tokens_by_sentence(text: str, token_budget: int, pattern: str, from_tail: bool) -> str:
    """
    Trim text to token_budget but try not to cut in the middle of a sentence.
    Mirrors RagFlow's sentence-aware context behavior.
    """
    text = remove_position_tag(text)
    if token_budget <= 0 or not text.strip():
        return ""
    sents = _split_sentences(text, pattern)
    if not sents:
        return _take_last_tokens(text, token_budget) if from_tail else _take_first_tokens(text, token_budget)
    collected: List[str] = []
    remaining = token_budget
    seq = reversed(sents) if from_tail else sents
    for s in seq:
        tks = num_tokens(s)
        if tks <= 0:
            continue
        if tks > remaining:
            # avoid cutting inside a sentence: if nothing collected yet, take this
            # whole sentence once; otherwise stop here.
            if not collected:
                collected.append(s)
            break
        collected.append(s)
        remaining -= tks
        if remaining <= 0:
            break
    if from_tail:
        collected = list(reversed(collected))
    return "".join(collected).strip()


def merge_sections(
    sections: List[Tuple[str, str]],
    config: ChunkingConfig,
) -> List[str]:
    """
    Merge sections into chunks by token limit and delimiter.
    sections: list of (text, position_tag).
    Returns list of chunk strings (each may contain position tags for PDF).
    """
    if not sections:
        return []
    cks: List[str] = []
    tk_nums: List[int] = []
    delim = config.delimiter
    chunk_max = config.chunk_token_num
    overlap_pct = max(0.0, min(100.0, config.overlapped_percent))
    custom = list(config.custom_delimiters) or []
    for m in re.finditer(r"`([^`]+)`", delim):
        custom.append(m.group(1))
    custom_pattern = "|".join(re.escape(t) for t in sorted(set(custom), key=len, reverse=True)) if custom else None

    def add_chunk(t: str, pos: str) -> None:
        nonlocal cks, tk_nums
        clean = remove_position_tag(t)
        tnum = num_tokens(clean)
        if tnum < config.min_chunk_tokens:
            pos = ""
        threshold = chunk_max * (100 - overlap_pct) / 100.0 if overlap_pct else chunk_max
        if not cks or tk_nums[-1] > threshold:
            if cks and overlap_pct > 0:
                prev_clean = remove_position_tag(cks[-1])
                start = int(len(prev_clean) * (100 - overlap_pct) / 100.0)
                t = prev_clean[start:] + t
            if pos and t.find(pos) < 0:
                t += pos
            cks.append(t)
            tk_nums.append(num_tokens(remove_position_tag(t)))
        else:
            if pos and cks[-1].find(pos) < 0:
                t = t + pos
            cks[-1] += t
            tk_nums[-1] += num_tokens(remove_position_tag(t))

    if custom_pattern:
        # When a custom delimiter matches (e.g. "### "), we treat it as the START
        # of the next chunk (heading begins the following chunk), matching the
        # visual structure of markdown and RAGFlow's Title chunker semantics.
        for sec, pos in sections:
            parts = re.split(r"(%s)" % custom_pattern, sec, flags=re.DOTALL)
            cur = ""
            for p in parts:
                if not p:
                    continue
                if re.fullmatch(custom_pattern, p):
                    # Flush accumulated text BEFORE the delimiter.
                    if cur.strip():
                        text = "\n" + cur if not cur.startswith("\n") else cur
                        if num_tokens(remove_position_tag(text)) < config.min_chunk_tokens:
                            pos = ""
                        if pos and text.find(pos) < 0:
                            text += pos
                        cks.append(text)
                        tk_nums.append(num_tokens(remove_position_tag(text)))
                    # Start a new chunk with the delimiter (heading marker).
                    cur = p
                else:
                    cur += p

            if cur.strip():
                text = "\n" + cur if not cur.startswith("\n") else cur
                if num_tokens(remove_position_tag(text)) < config.min_chunk_tokens:
                    pos = ""
                if pos and text.find(pos) < 0:
                    text += pos
                cks.append(text)
                tk_nums.append(num_tokens(remove_position_tag(text)))
        return cks

    for sec, pos in sections:
        add_chunk("\n" + sec if sec and not sec.startswith("\n") else sec, pos)
    return cks


def chunk_tables(
    tables: List[Tuple[Tuple[Optional[Any], str], List]],
    table_context_size: int = 0,
    image_context_size: int = 0,
    sections_for_context: Optional[List[Tuple[str, str]]] = None,
    attach_media_context: bool = False,
) -> List[Tuple[str, dict]]:
    """
    Each table/image becomes one chunk.

    If sections_for_context is provided and *_context_size > 0, we DO NOT mix
    the context text into page_content; instead we attach truncated context to
    metadata as `context_before` / `context_after` so that the core table/image
    content stays clean.
    """
    result: List[Tuple[str, dict]] = []
    for idx, ((_img, rows), _poss) in enumerate(tables):
        # Detect media type (DeepDOC-like): image blocks arrive as {"type": "image", "path": "..."}
        meta: dict
        if isinstance(_img, dict) and _img.get("type") == "image":
            meta = {"chunk_type": "image", "image_index": idx, "image_path": _img.get("path", "")}
        else:
            meta = {"chunk_type": "table", "table_index": idx}

        if attach_media_context:
            meta["attach_media_context"] = True

        if isinstance(rows, list):
            content = "\n".join(rows)
        else:
            content = rows or ""

        # Allow empty content for image chunks (content can be caption)
        if meta.get("chunk_type") != "image" and not content.strip():
            continue

        # Attach optional context into metadata (not into content).
        context_above_below = (
            table_context_size if meta.get("chunk_type") == "table" else image_context_size
        )
        if context_above_below and sections_for_context:
            above_parts: List[str] = []
            below_parts: List[str] = []
            for i in range(len(sections_for_context) - 1, -1, -1):
                t, _ = sections_for_context[i]
                t = remove_position_tag(t)
                if not t.strip():
                    continue
                if num_tokens("\n".join(above_parts) + t) <= context_above_below:
                    above_parts.insert(0, t)
                else:
                    break
            for i in range(len(sections_for_context)):
                t, _ = sections_for_context[i]
                t = remove_position_tag(t)
                if not t.strip():
                    continue
                if num_tokens("\n".join(below_parts) + t) <= context_above_below:
                    below_parts.append(t)
                else:
                    break
            if above_parts:
                meta["context_before"] = "\n\n".join(above_parts)
            if below_parts:
                meta["context_after"] = "\n\n".join(below_parts)
        result.append((content, meta))
    return result


def run_chunking(
    sections: List[Tuple[str, str]],
    tables: List[Tuple[Tuple[Optional[Any], str], List]],
    config: ChunkingConfig,
    source_name: str = "document",
) -> List[dict]:
    """
    Apply chunking strategies and return list of chunk dicts for RAG/data_processing.
    Each dict: {"page_content": str, "metadata": {"source": str, "chunk_index": int, "chunk_type": "text"|"table", ...}}
    """
    chunks: List[dict] = []
    text_chunks = merge_sections(sections, config)
    for content in text_chunks:
        clean = remove_position_tag(content)
        if not clean.strip():
            continue
        chunks.append(
            {
                "page_content": clean,
                "metadata": {
                    "source": source_name,
                    "chunk_index": len(chunks),
                    "chunk_type": "text",
                },
            }
        )
    table_chunks = chunk_tables(
        tables,
        table_context_size=config.table_context_size,
        image_context_size=config.image_context_size,
        sections_for_context=sections if (config.table_context_size or config.image_context_size) else None,
        attach_media_context=config.attach_media_context,
    )
    for content, meta in table_chunks:
        meta["source"] = source_name
        meta["chunk_index"] = len(chunks)
        chunks.append({"page_content": content, "metadata": meta})
    for i, c in enumerate(chunks):
        c["metadata"]["chunk_index"] = i
    return chunks


def run_chunking_blocks(
    blocks: List[dict],
    config: ChunkingConfig,
    source_name: str = "document",
) -> List[dict]:
    """
    Chunking solution that preserves original markdown order.

    blocks: ordered list of {"type": "text"|"table"|"image", ...}
    - contiguous text blocks are merged with token limits + overlap
    - table/image blocks become standalone chunks at their positions
    - optional table_context_size/image_context_size are attached from adjacent text
    """
    out: List[dict] = []
    text_buffer: List[Tuple[str, str]] = []  # (text, position_tag)
    prev_text_raw: str = ""
    last_text_chunk_index: Optional[int] = None
    sentence_pat = config.children_delimiters_pattern or r"([。！？!?；;\n])"

    def flush_text_buffer() -> None:
        nonlocal text_buffer, out, prev_text_raw, last_text_chunk_index
        if not text_buffer:
            return
        text_chunks = merge_sections(text_buffer, config)
        for content in text_chunks:
            clean = remove_position_tag(content)
            if not clean.strip():
                continue
            out.append(
                {
                    "page_content": clean,
                    "metadata": {
                        "source": source_name,
                        "chunk_index": len(out),
                        "chunk_type": "text",
                    },
                }
            )
        prev_text_raw = "\n\n".join(t for t, _ in text_buffer)
        text_buffer = []
        last_text_chunk_index = len(out) - 1 if out else None

    def next_text_raw_from(idx: int) -> str:
        parts: List[str] = []
        for j in range(idx + 1, len(blocks)):
            bt = blocks[j].get("type")
            if bt == "text":
                txt = (blocks[j].get("text") or "").strip()
                if txt:
                    parts.append(txt)
            elif bt in ("table", "image"):
                break
        return "\n\n".join(parts)

    for i, b in enumerate(blocks):
        btype = b.get("type")
        if btype == "text":
            txt = (b.get("text") or "").strip()
            if txt:
                text_buffer.append((txt, ""))
            continue

        if btype not in ("table", "image"):
            continue

        # Media boundary: flush preceding text chunks first.
        flush_text_buffer()

        above = ""
        below = ""
        next_txt = next_text_raw_from(i)

        if btype == "table" and config.table_context_size:
            above = _trim_to_tokens_by_sentence(prev_text_raw, config.table_context_size, sentence_pat, from_tail=True)
            below = _trim_to_tokens_by_sentence(next_txt, config.table_context_size, sentence_pat, from_tail=False)
        if btype == "image" and config.image_context_size:
            above = _trim_to_tokens_by_sentence(prev_text_raw, config.image_context_size, sentence_pat, from_tail=True)
            below = _trim_to_tokens_by_sentence(next_txt, config.image_context_size, sentence_pat, from_tail=False)

        if btype == "table":
            core = (b.get("text") or "").strip()
            if not core:
                continue
            meta = {
                "source": source_name,
                "chunk_index": len(out),
                "chunk_type": "table",
            }
        else:
            core = (b.get("caption") or "").strip()
            meta = {
                "source": source_name,
                "chunk_index": len(out),
                "chunk_type": "image",
                "image_path": b.get("path", ""),
            }

        if config.attach_media_context:
            meta["attach_media_context"] = True
        if above.strip():
            meta["context_before"] = above.strip()
        if below.strip():
            meta["context_after"] = below.strip()

        # Option D: attach image references into nearest preceding text chunk
        if btype == "image" and config.attach_images_to_section and last_text_chunk_index is not None:
            tmeta = out[last_text_chunk_index]["metadata"]
            imgs = tmeta.get("images") or []
            imgs.append({"path": meta.get("image_path", ""), "caption": core})
            tmeta["images"] = imgs
            continue

        out.append({"page_content": core, "metadata": meta})

    # Flush trailing text
    flush_text_buffer()

    # Option C: hierarchical merge (markdown headings) on text chunks
    if config.hierarchical_merge:
        out = hierarchical_merge_markdown(out, config, source_name=source_name)

    # Option B: parent-child chunking for retrieval
    if config.enable_child_chunks and config.include_children_in_output:
        out = attach_child_chunks(out, config, source_name=source_name)

    for idx, c in enumerate(out):
        c["metadata"]["chunk_index"] = idx
    return out


def attach_child_chunks(chunks: List[dict], config: ChunkingConfig, source_name: str) -> List[dict]:
    pattern = config.children_delimiters_pattern or r"([。！？!?；;\n])"
    try:
        compiled = re.compile(pattern, flags=re.DOTALL)
    except re.error:
        compiled = re.compile(r"([。！？!?；;\n])", flags=re.DOTALL)

    out: List[dict] = []
    for ck in chunks:
        out.append(ck)
        if ck.get("metadata", {}).get("chunk_type") != "text":
            continue
        mom_text = ck["page_content"]
        # split and keep delimiters, then re-join pairs
        parts = [p for p in compiled.split(mom_text) if p is not None]
        children: List[str] = []
        buf = ""
        for p in parts:
            if not p:
                continue
            if compiled.fullmatch(p):
                buf += p
                if buf.strip():
                    children.append(buf.strip())
                buf = ""
            else:
                buf += p
        if buf.strip():
            children.append(buf.strip())

        # Merge children that are too short to avoid trivial chunks like '- Bước 1:'
        merged_children: List[str] = []
        i = 0
        while i < len(children):
            s = children[i].strip()
            if not s:
                i += 1
                continue
            tks = num_tokens(s)
            # If this sentence is too short and there is a following sentence,
            # merge forward so that marker lines like '- Bước 1:' join with
            # their content.
            if tks < config.min_chunk_tokens and i + 1 < len(children):
                s = s + " " + children[i + 1].strip()
                i += 2
            else:
                i += 1

            # If previous merged child is still short, merge into it instead
            # of emitting a separate tiny child.
            if merged_children and num_tokens(merged_children[-1]) < config.min_chunk_tokens:
                merged_children[-1] = merged_children[-1].rstrip() + " " + s
            else:
                merged_children.append(s)

        for j, child_text in enumerate(merged_children):
            if not child_text.strip():
                continue
            out.append(
                {
                    "page_content": child_text,
                    "metadata": {
                        "source": source_name,
                        "chunk_index": -1,
                        "chunk_type": "child",
                        "mom_index": ck["metadata"]["chunk_index"],
                        "child_index": j,
                        "mom_text": mom_text,
                    },
                }
            )
    return out


def hierarchical_merge_markdown(chunks: List[dict], config: ChunkingConfig, source_name: str) -> List[dict]:
    """
    Simple markdown outline merge:
    - detect heading lines starting with '#'
    - group subsequent text chunks under the most recent heading path
    - then re-apply token-based merge to cap size
    """
    depth = max(1, int(config.hierarchical_depth or 3))
    heading_re = re.compile(r"^(#{1,6})\\s+")

    grouped: List[Tuple[str, str]] = []
    current_path: List[str] = []
    current_levels: List[int] = []
    buf: List[str] = []

    def flush_group():
        nonlocal buf
        txt = "\n".join([t for t in buf if t.strip()]).strip()
        if txt:
            grouped.append((txt, ""))
        buf = []

    def path_prefix() -> str:
        # keep only first N headings as context prefix
        if not current_path:
            return ""
        return "\n".join(current_path[-depth:]).strip()

    # only apply to text chunks; keep media chunks as barriers
    out: List[dict] = []
    for ck in chunks:
        if ck["metadata"].get("chunk_type") != "text":
            flush_group()
            # push any grouped text as merged text chunks
            if grouped:
                merged = merge_sections(grouped, config)
                for t in merged:
                    t = remove_position_tag(t)
                    if t.strip():
                        out.append(
                            {
                                "page_content": t,
                                "metadata": {
                                    "source": source_name,
                                    "chunk_index": len(out),
                                    "chunk_type": "text",
                                },
                            }
                        )
                grouped = []
            out.append(ck)
            continue

        text = ck["page_content"].strip()
        m = heading_re.match(text)
        if m:
            flush_group()
            lvl = len(m.group(1))
            while current_levels and current_levels[-1] >= lvl:
                current_levels.pop()
                if current_path:
                    current_path.pop()
            current_levels.append(lvl)
            current_path.append(text)
            # start a new group with heading as first line
            buf.append(text)
            continue

        # prepend path prefix once at the start of a new group
        if not buf and path_prefix():
            buf.append(path_prefix())
        buf.append(text)

    flush_group()
    if grouped:
        merged = merge_sections(grouped, config)
        for t in merged:
            t = remove_position_tag(t)
            if t.strip():
                out.append(
                    {
                        "page_content": t,
                        "metadata": {
                            "source": source_name,
                            "chunk_index": len(out),
                            "chunk_type": "text",
                        },
                    }
                )
    return out


