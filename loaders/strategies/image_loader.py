from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredImageLoader

def load_image(path: str, mode: str="elements") -> List[Document]:
    docs = UnstructuredImageLoader(path, mode=mode).load()
    for d in docs:
        d.metadata.setdefault("filetype", "image")
    return docs
