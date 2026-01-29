"""Image processing utilities for Ripperdoc."""

import base64
import mimetypes
from pathlib import Path
from typing import Optional, Tuple

from ripperdoc.utils.log import get_logger

logger = get_logger()

# Supported image formats
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
SUPPORTED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp"}

MAX_IMAGE_SIZE_BYTES = 32 * 1024 * 1024  # 32MB


def is_image_file(file_path: Path) -> bool:
    """Check if a file is a supported image format.

    Args:
        file_path: Path to the file

    Returns:
        True if the file has a supported image extension
    """
    return file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def detect_mime_type(file_path: Path) -> str:
    """Detect the MIME type of an image file.

    Args:
        file_path: Path to the image file

    Returns:
        MIME type string (e.g., "image/jpeg", "image/png")
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type in SUPPORTED_IMAGE_MIME_TYPES:
        return mime_type

    # Fallback to extension-based detection
    ext = file_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".gif":
        return "image/gif"
    if ext == ".webp":
        return "image/webp"
    if ext == ".bmp":
        return "image/bmp"
    return "image/jpeg"  # Default fallback


def read_image_as_base64(file_path: Path) -> Optional[Tuple[str, str]]:
    """Read an image file and return its base64-encoded data and MIME type.

    Args:
        file_path: Absolute path to the image file

    Returns:
        (base64_data, mime_type) tuple or None if reading fails
    """
    if not file_path.exists():
        logger.warning(
            "[image_utils] Image file not found",
            extra={"path": str(file_path)},
        )
        return None

    if not file_path.is_file():
        logger.warning(
            "[image_utils] Not a file",
            extra={"path": str(file_path)},
        )
        return None

    # Check file size
    file_size = file_path.stat().st_size
    if file_size > MAX_IMAGE_SIZE_BYTES:
        logger.warning(
            "[image_utils] Image too large",
            extra={
                "path": str(file_path),
                "size_bytes": file_size,
                "max_bytes": MAX_IMAGE_SIZE_BYTES,
            },
        )
        return None

    if not is_image_file(file_path):
        logger.warning(
            "[image_utils] Not a supported image format",
            extra={"path": str(file_path)},
        )
        return None

    try:
        with open(file_path, "rb") as f:
            image_bytes = f.read()

        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = detect_mime_type(file_path)

        logger.debug(
            "[image_utils] Loaded image",
            extra={
                "path": str(file_path),
                "size_bytes": len(image_bytes),
                "mime_type": mime_type,
            },
        )

        return (base64_data, mime_type)

    except (OSError, IOError) as e:
        logger.error(
            "[image_utils] Failed to read image",
            extra={"path": str(file_path), "error": str(e)},
        )
        return None
