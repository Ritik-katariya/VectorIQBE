from typing import List, Literal
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, PDFPlumberLoader

def load_pdf(path: str, strategy: Literal["auto","text","table"]="auto", extract_images: bool=False) -> List[Document]:
    if strategy == "text":
        docs = PyPDFLoader(path).load()
    elif strategy == "table":
        docs = PDFPlumberLoader(path, extract_images=False).load()
    else:
        docs = PyMuPDFLoader(path, extract_images=extract_images).load()
    for d in docs:
        d.metadata.setdefault("filetype", "pdf")
    return docs
