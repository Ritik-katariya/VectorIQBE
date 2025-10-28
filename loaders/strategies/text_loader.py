from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, UnstructuredFileLoader

TEXT_EXTS = {"txt","md","rst","csv","tsv","json","yaml","yml"}
DOC_EXTS  = {"docx","pptx","html","htm","eml"}

def load_textlike(path: str, encoding: str="utf-8") -> List[Document]:
    docs = TextLoader(path, encoding=encoding).load()
    for d in docs:
        d.metadata.setdefault("filetype", "text")
    return docs

def load_doclike_unstructured(path: str) -> List[Document]:
    docs = UnstructuredFileLoader(path).load()
    for d in docs:
        d.metadata.setdefault("filetype", "document")
    return docs
