
from langchain_core.documents import Document
from typing import Literal
from .strategies.pdf_loader import load_pdf
from .strategies.image_loader import  load_image_ocr
from .strategies.text_loader import load_textlike, load_doclike_unstructured, TEXT_EXTS, DOC_EXTS
from .strategies.web_loader import load_web_url, load_sitemap
from .strategies.fallback_loader import load_any
from utils.detect import sniff_bytes

def _read_head(path: str, n: int=12) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)

def load_to_documents(
    *,
    source_type: str,           # "file" | "url" | "text"
    path: str | None = None,    # for files
    filename: str | None = None,
    url: str | None = None,     # for a single URL or sitemap
    text: str | None = None,    # inline text
    pdf_strategy:Literal["auto", "text", "table"] = "auto",
    sitemap: bool = False,
    source_label: str | None = None,
) -> tuple[list[Document], str]:
    """
    Returns (docs, strategy_name)
    Creates LangChain Documents ONCE. No re-conversion later.
    """

    if source_type == "url":
        assert url, "url required"
        if sitemap:
            docs = load_sitemap(url, 200)
            strategy = "sitemap"
        else:
            docs = load_web_url([url])
            strategy = "web"
        for d in docs:
            d.metadata.setdefault("source", source_label or url)
        return docs, strategy

    if source_type == "text":
        assert text is not None, "text required"
        doc = Document(page_content=text, metadata={"filetype": "text", "source": source_label or "inline"})
        return [doc], "text-inline"

    # files
    assert path, "path required for file"
    ext = (filename or "").lower().rsplit(".", 1)[-1] if filename and "." in filename else ""
    if ext == "pdf":
        docs = load_pdf(path, pdf_strategy, False)
        strategy = f"pdf:{pdf_strategy}"
    elif ext in {"png","jpg","jpeg","gif","bmp","tiff","webp"}:
        docs = load_image_ocr(path)
        strategy = "image"
    elif ext in TEXT_EXTS:
        docs = load_textlike(path)
        strategy = "text"
    elif ext in DOC_EXTS:
        docs = load_doclike_unstructured(path)
        strategy = "doclike"
    else:
        # sniff header
        head = _read_head(path, 12)
        kind = sniff_bytes(head) or ""
        if kind == "pdf":
            docs = load_pdf(path, pdf_strategy, False)
            strategy = f"pdf:{pdf_strategy}"
        elif kind.startswith("image/"):
            docs = load_image_ocr( path)
            strategy = "image"
        else:
            docs = load_any(path)
            strategy = "fallback"
    for d in docs:
        d.metadata.setdefault("source", source_label or (filename or path))
    return docs, strategy
