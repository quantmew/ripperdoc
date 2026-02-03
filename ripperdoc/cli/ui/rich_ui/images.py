"""Image handling helpers for the Rich UI."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ripperdoc.cli.ui.helpers import get_profile_for_pointer
from ripperdoc.core.config import model_supports_vision
from ripperdoc.utils.image_utils import is_image_file, read_image_as_base64
from ripperdoc.utils.log import get_logger

logger = get_logger()


def _extract_image_paths(text: str) -> List[str]:
    """Extract @-referenced image paths from text.

    Handles cases like:
    - "@image.png describe this" (space after)
    - "@image.png描述这个" (no space after, Chinese text)
    - "@image.png.whatIsThis" (no space, ASCII text)

    Args:
        text: User input text

    Returns:
        List of file paths (without the @ prefix)
    """
    result = []

    # Find all @ followed by content until space or end
    for match in re.finditer(r"@(\S+)", text):
        candidate = match.group(1)
        if not candidate:
            continue

        # Try to find the actual file path by progressively trimming
        # First, check if the full candidate is a file
        if Path(candidate).exists():
            result.append(candidate)
            continue

        # Not a file, try to find where the file path ends
        # Common file extensions
        extensions = [
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".webp",
            ".bmp",
            ".svg",
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".vue",
            ".go",
            ".rs",
            ".java",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".cs",
            ".php",
            ".rb",
            ".sh",
            ".md",
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            ".xml",
            ".html",
            ".css",
            ".scss",
            ".sql",
            ".db",
        ]

        found_path = None
        for ext in extensions:
            # Look for this extension in the candidate
            if ext.lower() in candidate.lower():
                # Found extension, extract path up to and including it
                ext_pos = candidate.lower().find(ext.lower())
                potential_path = candidate[: ext_pos + len(ext)]
                if Path(potential_path).exists():
                    found_path = potential_path
                    break

                # Also try to find the LAST occurrence of this extension
                # For cases like "file.txt.extraText"
                last_ext_pos = candidate.lower().rfind(ext.lower())
                if last_ext_pos > ext_pos:
                    potential_path = candidate[: last_ext_pos + len(ext)]
                    if Path(potential_path).exists():
                        found_path = potential_path
                        break

        if found_path:
            result.append(found_path)
        else:
            # No file found, keep the original candidate
            # The processing function will handle non-existent files
            result.append(candidate)

    return result


def process_images_in_input(
    user_input: str,
    project_path: Path,
    model_pointer: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Process @ references for images in user input.

    Only image files are processed and converted to image blocks.
    Text files and non-existent files are left as-is in the text.

    Args:
        user_input: Raw user input text
        project_path: Project root path
        model_pointer: Model pointer to check vision support

    Returns:
        (processed_text, image_blocks) tuple
    """
    image_blocks: List[Dict[str, Any]] = []
    processed_text = user_input

    # Check if current model supports vision
    profile = get_profile_for_pointer(model_pointer)
    supports_vision = profile and model_supports_vision(profile)

    if not supports_vision:
        # Model doesn't support vision, leave all @ references as-is
        return processed_text, image_blocks

    referenced_paths = _extract_image_paths(user_input)

    for ref_path in referenced_paths:
        # Try relative path first, then absolute path
        path_candidate = project_path / ref_path
        if not path_candidate.exists():
            path_candidate = Path(ref_path)

        if not path_candidate.exists():
            logger.debug(
                "[ui] @ referenced file not found",
                extra={"path": ref_path},
            )
            # Keep the reference in text (LLM should know file doesn't exist)
            continue

        # Only process image files
        if not is_image_file(path_candidate):
            # Not an image file, keep @ reference in text
            # The LLM can decide to read it with the Read tool if needed
            continue

        # Process image file
        result = read_image_as_base64(path_candidate)
        if result:
            base64_data, mime_type = result
            image_blocks.append(
                {
                    "type": "image",
                    "source_type": "base64",
                    "media_type": mime_type,
                    "image_data": base64_data,
                }
            )
            # Remove image reference from text (content included separately as image block)
            processed_text = processed_text.replace(f"@{ref_path}", "")
        else:
            # Failed to read image, keep reference in text
            logger.warning(
                "[ui] Failed to read @ referenced image",
                extra={"path": ref_path},
            )

    # Clean up extra whitespace
    processed_text = re.sub(r"\s+", " ", processed_text).strip()

    return processed_text, image_blocks
