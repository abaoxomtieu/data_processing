"""
Detect Markdown tables in a .md file (GFM pipe syntax: | col | col |).
Returns list of (start_line_1based, end_line_1based, raw_lines) for each table.
"""
import re
from pathlib import Path
from typing import List, Tuple


# Dòng separator của bảng GFM: chỉ chứa |, -, :, khoảng trắng, ví dụ |---|---|
TABLE_SEP_PATTERN = re.compile(r"^\|[\s\-:]+\|$")


def is_table_separator(line: str) -> bool:
    """True nếu dòng là separator của bảng (|---|---|)."""
    line = line.strip()
    if not line or "|" not in line:
        return False
    # Cho phép | ở đầu/cuối tùy style
    return bool(TABLE_SEP_PATTERN.match(line)) or bool(
        re.match(r"^[\s\-:|]+\|?$", line) and line.count("|") >= 2
    )


def is_table_row(line: str) -> bool:
    """True nếu dòng trông giống một hàng bảng (có nhiều |)."""
    line = line.strip()
    return "|" in line and line.count("|") >= 2


def find_tables_in_markdown(content: str) -> List[Tuple[int, int, List[str]]]:
    """
    Tìm tất cả block bảng GFM trong nội dung markdown.
    Returns: list of (start_line_1based, end_line_1based, lines_of_table).
    """
    lines = content.splitlines()
    tables: List[Tuple[int, int, List[str]]] = []
    i = 0
    while i < len(lines):
        if not is_table_row(lines[i]):
            i += 1
            continue
        start = i + 1  # 1-based
        block = [lines[i]]
        i += 1
        has_sep = False
        while i < len(lines) and is_table_row(lines[i]):
            if is_table_separator(lines[i]):
                has_sep = True
            block.append(lines[i])
            i += 1
        if has_sep and len(block) >= 2:
            end = i  # 1-based end line
            tables.append((start, end, block))
    return tables


def find_tables_in_file(md_path: Path) -> List[Tuple[int, int, List[str]]]:
    """Đọc file .md và trả về danh sách bảng."""
    text = md_path.read_text(encoding="utf-8")
    return find_tables_in_markdown(text)


if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "output" / "output.md"
    tables = find_tables_in_file(path)
    print(f"Found {len(tables)} table(s) in {path}")
    for start, end, block in tables:
        print(f"  Lines {start}-{end} ({len(block)} rows):")
        for row in block[:3]:
            print(f"    {row[:80]}{'...' if len(row) > 80 else ''}")
        if len(block) > 3:
            print(f"    ... and {len(block) - 3} more rows")

