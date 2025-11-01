# sniffer.py
import os
import re
import sys
from typing import Optional

# Config: default OFF on Windows (can enable via env)
USE_LIBMAGIC: bool = os.getenv("SNIFFER_USE_LIBMAGIC", "").lower() in {"1", "true", "yes"}
if sys.platform == "win32":
    USE_LIBMAGIC = False  # keep off by default on Windows

PDF_HEADER = b"%PDF-"
IMAGE_MAGIC = {
    b"\xFF\xD8\xFF": "jpg",             # JPEG
    b"\x89PNG\r\n\x1a\n": "png",        # PNG
    b"GIF87a": "gif",                   # GIF87a
    b"GIF89a": "gif",                   # GIF89a
}
URL_RE = re.compile(r"^https?://", re.I)


def is_url(s: str) -> bool:
    return bool(URL_RE.match(s))


def _looks_like_text(b: bytes, min_ratio: float = 0.85) -> bool:
    if not b:
        return False
    s = b.decode("utf-8", errors="ignore")
    if not s:
        return False
    printable = sum(ch.isprintable() or ch.isspace() for ch in s)
    return (printable / max(1, len(s))) >= min_ratio


def sniff_bytes(file_bytes: bytes) -> Optional[str]:
    """
    Detect file type from raw bytes.
    Returns: "pdf" | "image/<ext>" | "text" | None
    """
    # 1) Fast signatures
    if file_bytes.startswith(PDF_HEADER):
        return "pdf"
    for sig, ext in IMAGE_MAGIC.items():
        if file_bytes.startswith(sig):
            return f"image/{ext}"

    # 2) Optional libmagic path (lazy import, guarded)
    if USE_LIBMAGIC:
        try:
            import magic  # import ONLY if we need it
            mime = (magic.from_buffer(file_bytes, mime=True) or "").lower()
            if "pdf" in mime:
                return "pdf"
            if mime.startswith("image/"):
                return mime
            if "text" in mime or "json" in mime:
                return "text"
        except Exception:
            # swallow libmagic issues and fall through
            pass

    # 3) Fallback heuristic
    if _looks_like_text(file_bytes):
        return "text"

    return None
