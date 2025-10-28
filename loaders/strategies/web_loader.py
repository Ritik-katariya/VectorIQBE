from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader, SitemapLoader

def load_web_url(urls: List[str]) -> List[Document]:
    loader = WebBaseLoader(urls)
    docs = loader.load()
    for d in docs:
        d.metadata.setdefault("filetype", "web")
    return docs

def load_sitemap(sitemap_url: str, max_docs: Optional[int]=200) -> List[Document]:
    loader = SitemapLoader(sitemap_url)
    docs = loader.load()
    if max_docs:
        docs = docs[:max_docs]
    for d in docs:
        d.metadata.setdefault("filetype", "web")
    return docs
