"""PDF reading utilities for Ripperdoc."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def parse_page_range(pages_str: str, total_pages: int) -> list[int]:
    """Parse a page range string into a list of 0-indexed page numbers.

    Accepts formats like "1-5", "3", "10-20". Pages are 1-indexed in input
    but returned as 0-indexed for internal use.

    Enforces a maximum of 20 pages per request.
    """
    result: list[int] = []
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            try:
                start = int(start_str.strip())
                end = int(end_str.strip())
            except ValueError:
                raise ValueError(f"Invalid page range: {part}")
            start = max(1, start)
            end = min(total_pages, end)
            result.extend(range(start - 1, end))
        else:
            try:
                page = int(part)
            except ValueError:
                raise ValueError(f"Invalid page number: {part}")
            if 1 <= page <= total_pages:
                result.append(page - 1)

    if len(result) > 20:
        raise ValueError(f"Too many pages requested ({len(result)}). Maximum 20 pages per request.")

    return sorted(set(result))


def read_pdf_text(
    file_path: Path,
    pages: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Read text from a PDF file.

    Args:
        file_path: Path to the PDF file.
        pages: Optional page range string (e.g., "1-5", "3", "10-20").

    Returns:
        Tuple of (extracted_text, error_message).
        On success: (text, None)
        On failure: ("", error_message)
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return (
            "",
            "PDF support requires the 'pypdf' package. Install it with: pip install pypdf",
        )

    if not file_path.exists():
        return "", f"PDF file not found: {file_path}"

    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:
        return "", f"Failed to open PDF: {exc}"

    total_pages = len(reader.pages)

    if total_pages == 0:
        return "", "PDF has no pages"

    # If 10+ pages and no page range specified, require pages parameter
    if total_pages >= 10 and pages is None:
        return (
            "",
            f"PDF has {total_pages} pages. Please specify a page range using the 'pages' parameter "
            f"(e.g., pages='1-5'). Maximum 20 pages per request.",
        )

    # Determine which pages to read
    if pages:
        try:
            page_indices = parse_page_range(pages, total_pages)
        except ValueError as exc:
            return "", str(exc)
    else:
        # Less than 10 pages, read all
        page_indices = list(range(total_pages))

    # Extract text
    text_parts: list[str] = []
    for idx in page_indices:
        try:
            page_text = reader.pages[idx].extract_text() or ""
            text_parts.append(f"--- Page {idx + 1} ---\n{page_text}")
        except Exception as exc:
            text_parts.append(f"--- Page {idx + 1} ---\n[Error extracting text: {exc}]")

    return "\n\n".join(text_parts), None
