from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from lib.chroma_connection import get_temporary_collection, get_chroma_client, get_permanent_collection
import os
from dotenv import load_dotenv

load_dotenv()  # Add this line to load environment variables

class SessionStore:
    """
    In-memory session store for temporary chunks.
    NOT for production persistence—attach Redis if needed.
    """
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.embed = OpenAIEmbeddings(
            model=model_name,
            headers={"User-Agent": os.getenv("USER_AGENT", "VectorIQ/0.1.0")}
        )
        self.client = get_chroma_client()

    def put(self, session_id: str, chunks: List[Document]) -> None:
        documents = [doc.page_content for doc in chunks]
        metadatas = [{"session_id": session_id, **doc.metadata} for doc in chunks]
        ids = [f"temp_{session_id}_{i}" for i in range(len(chunks))]
        
        embeddings = self.embed.embed_documents(documents)
        
        collection = get_temporary_collection()
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def get(self, session_id: str) -> List[Document]:
        collection = get_temporary_collection()
        results = collection.get(
            where={"session_id": session_id}
        )
        
        documents = []
        if results and results.get('documents'):
            for doc, metadata in zip(results['documents'], results['metadatas']):
                documents.append(Document(page_content=doc, metadata=metadata))
        return documents

    def clear(self, session_id: str) -> None:
        collection = get_temporary_collection()
        collection.delete(
            where={"session_id": session_id}
        )
