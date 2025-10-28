import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
import os
from typing import Dict

load_dotenv()

_client: ClientAPI | None = None
_permanent_collections: Dict[str, Collection] = {}
_temporary_collection: Collection | None = None

def get_chroma_client() -> ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE")
        )
    return _client

def get_permanent_collection(base_collection: str = "knowledge", namespace: str = None) -> Collection:
    global _permanent_collections
    client = get_chroma_client()
    coll_name = f"{base_collection}_{namespace}" if namespace else base_collection
    
    if coll_name not in _permanent_collections:
        _permanent_collections[coll_name] = client.get_or_create_collection(
            name=coll_name,
            metadata={"type": "permanent", "namespace": namespace}
        )
    return _permanent_collections[coll_name]

def get_temporary_collection() -> Collection:
    global _temporary_collection
    if _temporary_collection is None:
        client = get_chroma_client()
        _temporary_collection = client.get_or_create_collection(
            name="temporary_collection",
            metadata={"type": "temporary"}
        )

    return _temporary_collection