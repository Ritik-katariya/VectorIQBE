from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 120,
    min_chunk_chars: int = 1,          # set >1 to drop tiny chunks
    strip_whitespace: bool = True,
) -> List[Document]:
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,          # ensures 'metadata["start_index"]' on chunks
    )

    # Clean/prep input docs (avoid None content)
    prepped: List[Document] = []
    for d in docs:
        text = d.page_content or ""
        if strip_whitespace:
            # normalize common whitespace; keep newlines for semantic breaks
            text = text.replace("\r\n", "\n").strip()
        if not text:
            continue
        prepped.append(Document(page_content=text, metadata=dict(d.metadata or {})))

    # Let LangChain do the heavy lifting
    chunks: List[Document] = splitter.split_documents(prepped)

    # Add a stable chunk_index per original doc (based on source + optional page)
    out: List[Document] = []
    counters: dict[Optional[str], int] = {}
    for c in chunks:
        meta = dict(c.metadata or {})
        # Build a per-document key; tweak fields to your schema
        key = f'{meta.get("source")}:::{meta.get("page", meta.get("page_number"))}'
        counters[key] = counters.get(key, 0) + 1
        idx = counters[key] - 1

        if len(c.page_content) < min_chunk_chars:
            continue

        meta["chunk_index"] = idx
        # 'start_index' is already present if add_start_index=True
        out.append(Document(page_content=c.page_content, metadata=meta))

    return out
