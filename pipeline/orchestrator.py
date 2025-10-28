from typing import Dict, Any
from langchain_core.documents import Document
from loaders.general_loader import load_to_documents
from pipeline.chunker import chunk_documents
from stores.temp_store import SessionStore
from stores.permanent_store import PermanentVectorStore
from utils.types import LoadParams, ChunkParams, StoreChoice, PipelineResult

_temp_store = None
_perm_store = None

def _get_temp_store():
    global _temp_store
    if _temp_store is None:
        _temp_store = SessionStore()
    return _temp_store

def _get_perm_store():
    global _perm_store
    if _perm_store is None:
        _perm_store = PermanentVectorStore()
    return _perm_store

def run_pipeline(load: LoadParams, chunk: ChunkParams, store: StoreChoice) -> PipelineResult:
    # 1) Load → Documents (once)
    docs, strategy = load_to_documents(
        source_type=load.source_type,
        path=load.path,
        filename=(load.path.split("/")[-1] if load.path else None),
        url=load.url,
        text=load.text,
        pdf_strategy=load.pdf_strategy,
        sitemap=load.sitemap,
        source_label=load.source_label,
    )

    # 2) Chunk (Document -> Document)
    chunks = chunk_documents(docs, chunk_size=chunk.chunk_size, chunk_overlap=chunk.chunk_overlap)

    # 3) Store
    if store.mode == "temporary":
        assert store.session_id, "session_id required for temporary mode"
        # attach session metadata
        for c in chunks:
            c.metadata.update({"datastore": "temporary", "session_id": store.session_id})
        _get_temp_store().put(store.session_id, chunks)
    else:
        # attach namespace/user/org metadata
        for c in chunks:
            c.metadata.update({"datastore": "permanent", "namespace": store.namespace})
        collection = _get_perm_store().upsert(chunks, base_collection="knowledge", namespace=store.namespace)

    # response sample (no large payloads)
    sample = [{"content": c.page_content[:800], "metadata": c.metadata} for c in chunks[:5]]

    return PipelineResult(
        total_chunks=len(chunks),
        strategy=strategy,
        sample=sample,
    )
