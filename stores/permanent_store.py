from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from lib.chroma_connection import get_permanent_collection, get_chroma_client, get_temporary_collection
import os
from dotenv import load_dotenv

load_dotenv()  # Add this line to load environment variables

class PermanentVectorStore:
    """
    Local Chroma + OpenAI embeddings (requires OPENAI_API_KEY in env).
    Use `namespace` to separate tenants (stored in metadata & collection name suffix).
    """
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self._embed = None
        self._client = None
    
    @property
    def embed(self):
        if self._embed is None:
            # OpenAIEmbeddings reads OPENAI_API_KEY from environment
            self._embed = OpenAIEmbeddings(
                model=self.model_name,
                headers={"User-Agent": os.getenv("USER_AGENT", "VectorIQ/0.1.0")}
            )
        return self._embed
    
    @property
    def client(self):
        if self._client is None:
            self._client = get_chroma_client()
        return self._client

    def upsert(
        self,
        chunks: List[Document],
        base_collection: str = "knowledge",
        namespace: Optional[str] = None,
    ) -> str:
        documents = [doc.page_content for doc in chunks]
        metadatas = [{"namespace": namespace, **doc.metadata} for doc in chunks]
        ids = [f"doc_{i}_{namespace or 'default'}" for i in range(len(chunks))]
        
        embeddings = self.embed.embed_documents(documents)
        
        collection = get_permanent_collection(base_collection, namespace)
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        return collection.name
