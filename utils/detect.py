import re
import magic
PDF_HEADER = b"%PDF-"
IMAGE_MAGIC = {b"\xFF\xD8\xFF": "jpg", b"\x89PNG\r\n\x1a\n": "png", b"GIF87a": "gif", b"GIF89a": "gif"}
URL_RE = re.compile(r"^https?://", re.I)

def is_url(s: str) -> bool:
    return bool(URL_RE.match(s))

def sniff_bytes(file_bytes: bytes) -> str | None:
    if file_bytes.startswith(PDF_HEADER): return "pdf"
    for sig, ext in IMAGE_MAGIC.items():
        if file_bytes.startswith(sig): return f"image/{ext}"
    try:
        mime = magic.from_buffer(file_bytes, mime=True) or ""
        if "pdf" in mime: return "pdf"
        if mime.startswith("image/"): return mime
        if "text" in mime or "json" in mime: return "text"
    except Exception:
        pass
    return None
