from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import os, json, numpy as np
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from helpers.collections import get_existing_collection

load_dotenv()
router = APIRouter()

# ---------------- Request Model ----------------
class QueryRequest(BaseModel):
    ids: List[str]
    query: str
    namespace: Optional[str] = None
    base_collection: str = "knowledge"
    temperature: float = 0.3

# ---------------- Stream Query Response ----------------
async def stream_query_response(query: str, vector_ids: List[str], namespace: Optional[str],
                                base_collection: str, temperature: float):
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            headers={"User-Agent": os.getenv("USER_AGENT", "VectorIQ/0.1.0")}
        )

        # Get existing collection
        collection = get_existing_collection(base_collection, namespace)

        # Fetch only the given document IDs
        results = collection.get(
            ids=vector_ids,
            include=["documents", "metadatas", "embeddings"]
        )

        # Safely extract results, handling numpy arrays
        docs = results.get("documents")
        if docs is None:
            docs = []
        elif isinstance(docs, np.ndarray):
            if docs.size == 0:
                docs = []
            else:
                docs = docs.tolist()
        
        metadatas = results.get("metadatas")
        if metadatas is None:
            metadatas = []
        elif isinstance(metadatas, np.ndarray):
            if metadatas.size == 0:
                metadatas = []
            else:
                metadatas = metadatas.tolist()
        
        doc_embeds = results.get("embeddings")
        if doc_embeds is None:
            doc_embeds = []
        elif isinstance(doc_embeds, np.ndarray):
            if doc_embeds.size == 0:
                doc_embeds = []
            else:
                doc_embeds = doc_embeds.tolist()

        if len(docs) == 0:
            yield "data: " + json.dumps({
                "type": "error",
                "error": "No data found in given documents"
            }) + "\n\n"
            return

        # Compute query embedding
        query_emb = np.array(embeddings.embed_query(query), dtype=np.float32)
        qnorm = np.linalg.norm(query_emb)
        # Convert numpy scalar to Python float to avoid ambiguous truth value errors
        qnorm = float(qnorm.item() if hasattr(qnorm, 'item') else qnorm)
        if abs(qnorm) < 1e-10:
            qnorm = 1.0

        # Rank docs by cosine similarity
        sims = []
        for emb in doc_embeds:
            v = np.array(emb, dtype=np.float32)
            vnorm = np.linalg.norm(v)
            # Convert numpy scalar to Python float to avoid ambiguous truth value errors
            vnorm = float(vnorm.item() if hasattr(vnorm, 'item') else vnorm)
            if abs(vnorm) < 1e-10:
                vnorm = 1.0
            sims.append(float(np.dot(query_emb, v) / (qnorm * vnorm)))

        order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        ranked_docs = [docs[i] for i in order]
        ranked_meta = [metadatas[i] for i in order]
        ranked_ids = [vector_ids[i] for i in order]

        # Prepare context
        context = "\n\n".join([f"[Doc {i+1}]\n{ranked_docs[i]}" for i in range(len(ranked_docs))])
        sources = []
        for i in range(len(ranked_docs)):
            meta = ranked_meta[i] if i < len(ranked_meta) and ranked_meta[i] is not None else {}
            # Handle numpy arrays in metadata
            if isinstance(meta, np.ndarray):
                meta = meta.tolist() if meta.size > 0 else {}
            elif not isinstance(meta, dict):
                meta = {}
            sources.append({
                "id": ranked_ids[i],
                "source": meta.get("source", "Unknown source") if isinstance(meta, dict) else "Unknown source"
            })

        # Build LLM prompt
        system_prompt = (
            "You are a helpful assistant. Use only the provided context documents to answer clearly and accurately. "
            "If the answer cannot be found in the documents, say so."
        )

        user_prompt = (
            f"Question: {query}\n\n"
            f"Context Documents:\n{context}\n\n"
            "Give a concise, human-friendly answer using this context."
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature, streaming=True)

        # Send metadata first
        yield "data: " + json.dumps({
            "type": "metadata",
            "sources": sources,
            "total_chunks": len(ranked_docs)
        }) + "\n\n"

        # Stream LLM output
        full_resp = ""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        async for chunk in llm.astream(messages):
            content = getattr(chunk, "content", None)
            if content:
                full_resp += content
                yield "data: " + json.dumps({"type": "content", "content": content}) + "\n\n"

        yield "data: " + json.dumps({
            "type": "final",
            "summary": "Response completed",
            "total_tokens": len(full_resp.split())
        }) + "\n\n"

    except HTTPException as e:
        yield "data: " + json.dumps({"type": "error", "error": e.detail}) + "\n\n"
    except Exception as e:
        yield "data: " + json.dumps({"type": "error", "error": str(e)}) + "\n\n"

# ---------------- Route ----------------
@router.post("/query")
async def query_vectors(request: QueryRequest):
    if not request.ids or not request.query:
        raise HTTPException(status_code=400, detail="Both 'ids' and 'query' are required.")

    return StreamingResponse(
        stream_query_response(
            query=request.query.strip(),
            vector_ids=request.ids,
            namespace=request.namespace,
            base_collection=request.base_collection,
            temperature=request.temperature
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
