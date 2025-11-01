from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import os, tempfile, shutil
from utils.types import LoadParams, ChunkParams, StoreChoice
from pipeline.orchestrator import run_pipeline
from typing import Literal


router = APIRouter()

def _save_temp(upload: UploadFile) -> str:
    suffix = ""
    if upload.filename and "." in upload.filename: suffix = "." + upload.filename.rsplit(".",1)[-1]
    fd, path = tempfile.mkstemp(suffix=suffix); os.close(fd)
    with open(path, "wb") as f: shutil.copyfileobj(upload.file, f)
    return path

@router.post("/ingest")
def ingest(
    # choose exactly one of these:
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
    text: str | None = Form(None),

    # hints
    pdf_strategy: Literal["auto", "text", "table"] = "auto",
    sitemap: bool = Form(False),
    source_label: str | None = Form(None),

    # chunking
    chunk_size: int = Form(900),
    chunk_overlap: int = Form(120),

    # storage
    store_mode: str = Form("temporary"),          # "temporary" | "permanent"
    session_id: str | None = Form(None),
    namespace: str | None = Form(None),
):
    provided = [x is not None for x in (file, url, text)]
    if sum(provided) != 1:
        raise HTTPException(400, "Provide exactly one of: file, url, or text")

    # build LoadParams
    if file:
        path = _save_temp(file)
        lp = LoadParams(source_type="file", path=path, pdf_strategy=pdf_strategy, sitemap=False, source_label=source_label)
    elif url:
        lp = LoadParams(source_type="url", url=url, pdf_strategy=pdf_strategy, sitemap=sitemap, source_label=source_label)
    else:
        lp = LoadParams(source_type="text", text=text, pdf_strategy=pdf_strategy, source_label=source_label)

    cp = ChunkParams(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    sc = StoreChoice(mode=store_mode, session_id=session_id, namespace=namespace, metadata=None)

    try:
        result = run_pipeline(lp, cp, sc)
        return result.dict()
    finally:
        if file:
            try: os.remove(lp.path)  # cleanup temp file
            except Exception: pass

@router.get("/session/{session_id}")
def get_session(session_id: str):
    from stores.temp_store import SessionStore
    store = SessionStore()
    docs = store.get(session_id)
    return {"chunks": len(docs)}

@router.get("/status")
async def data_loader_status():
    return {"data_loader": "ok"}
