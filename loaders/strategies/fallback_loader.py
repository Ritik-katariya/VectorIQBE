from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredFileLoader

def load_any(path: str) -> List[Document]:
    docs = UnstructuredFileLoader(path).load()
    for d in docs:
        d.metadata.setdefault("filetype", "unknown")
    return docs
