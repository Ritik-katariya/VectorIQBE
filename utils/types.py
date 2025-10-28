from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel
from langchain_core.documents import Document

SourceType = Literal["file", "url", "text"]

class LoadParams(BaseModel):
    source_type: SourceType
    # exactly one of:
    path: Optional[str] = None      # for files
    url: Optional[str] = None       # for single URL or sitemap
    text: Optional[str] = None      # inline text
    # loader hints
    pdf_strategy: Literal["auto","text","table"] = "auto"
    sitemap: bool = False
    source_label: Optional[str] = None  # for metadata "source"

class ChunkParams(BaseModel):
    chunk_size: int = 900
    chunk_overlap: int = 120

class StoreChoice(BaseModel):
    mode: Literal["temporary","permanent"]
    session_id: Optional[str] = None     # required if mode=temporary
    namespace: Optional[str] = None      # used for permanent vector DB
    metadata: Optional[Dict[str, Any]] = None  # user_id/org_id etc.

class PipelineResult(BaseModel):
    total_chunks: int
    strategy: str
    sample: List[Dict[str, Any]]
