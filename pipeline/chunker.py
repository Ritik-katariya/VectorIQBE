from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(
    docs: List[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    out: List[Document] = []
    for d in docs:
        meta = dict(d.metadata or {})
        for i, part in enumerate(splitter.split_text(d.page_content)):
            out.append(Document(page_content=part, metadata={**meta, "chunk_index": i}))
    return out
