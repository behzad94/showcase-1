# src/api/server.py
# goal: FastAPI service with /rebuild_index and /ask endpoints
# we lazily load persisted FAISS and auto-rebuild if corpus changed.

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.ingest.ingest import scan_corpus_dir
from src.chunk.chunker import chunk_corpus, Chunk
from src.embed.embedder import embed_chunks, _model as _EMBED_MODEL
from src.store.vectordb import VectorDB
from src.search.retriever import Retriever
from src.answer.rag import RagAnswerer
from src.utils.audit import log_event

DATA_DIR = os.environ.get("RAG_DATA_DIR", "data/processed")
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")
DEFAULT_CORPUS = os.environ.get("RAG_CORPUS_DIR", "corpus")

app = FastAPI(title="RAG Showcase API")

db: Optional[VectorDB] = None
retriever: Optional[Retriever] = None
rag: Optional[RagAnswerer] = None
chunks: Optional[List[Chunk]] = None

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    summarize: bool = False

def _corpus_changed(corpus_path: str) -> bool:
    corpus = Path(corpus_path)
    if not corpus.exists():
        return False
    latest_mod = 0
    for f in corpus.iterdir():
        if f.is_file():
            latest_mod = max(latest_mod, f.stat().st_mtime)
    last_index_build = os.path.getmtime(MANIFEST_PATH) if os.path.exists(MANIFEST_PATH) else 0
    return latest_mod > last_index_build

def _load_or_chunk_corpus(corpus_path: str) -> List[Chunk]:
    docs = scan_corpus_dir(Path(corpus_path))
    if not docs:
        raise HTTPException(status_code=400, detail=f"No documents found in '{corpus_path}/'")
    chs = chunk_corpus(docs, chunk_tokens=500, overlap_tokens=50)
    if not chs:
        raise HTTPException(status_code=400, detail="No chunks produced from documents")
    return chs

def _ensure_ready(corpus_path: str = DEFAULT_CORPUS) -> None:
    global db, retriever, rag, chunks
    if _corpus_changed(corpus_path):
        rebuild_index(corpus_path)
        return
    if db is not None and retriever is not None and rag is not None and chunks:
        return
    try:
        loaded_db = VectorDB.load(INDEX_PATH, MANIFEST_PATH, metric="ip")
        chs = _load_or_chunk_corpus(corpus_path)
        retr = Retriever(loaded_db, chs)
        ra = RagAnswerer(retr)
        db, chunks, retriever, rag = loaded_db, chs, retr, ra
        log_event("persist_load_ok", {
            "index_path": INDEX_PATH, "manifest_path": MANIFEST_PATH,
            "dim": loaded_db.dim, "embed_model": getattr(_EMBED_MODEL, "model_name_or_path", "all-MiniLM-L6-v2"),
            "chunks": len(chs), "ntotal": int(loaded_db.index.ntotal),
        })
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail="Index not built yet. Use /rebuild_index first.") from e
    except Exception as e:
        log_event("persist_load_error", {"error": str(e)})
        raise HTTPException(status_code=500, detail="Failed to load persisted index")

@app.post("/rebuild_index")
def rebuild_index(corpus_path: str = DEFAULT_CORPUS):
    global db, retriever, rag, chunks
    try:
        docs = scan_corpus_dir(Path(corpus_path))
        if not docs:
            log_event("rebuild_index_failed", {"corpus_path": corpus_path, "reason": "empty_corpus"})
            raise HTTPException(status_code=400, detail=f"No documents found in '{corpus_path}/'")
        chunks = chunk_corpus(docs, 500, 50)
        embs = embed_chunks(chunks)
        if not embs:
            log_event("rebuild_index_failed", {"corpus_path": corpus_path, "reason": "no_embeddings"})
            raise HTTPException(status_code=500, detail="Embedding failed")
        os.makedirs(DATA_DIR, exist_ok=True)
        db = VectorDB(dim=embs[0].vector.shape[0], metric="ip")
        db.add_embeddings(embs)
        db.save(index_path=INDEX_PATH, manifest_path=MANIFEST_PATH,
                embed_model=getattr(_EMBED_MODEL, "model_name_or_path", "all-MiniLM-L6-v2"))
        retriever = Retriever(db, chunks)
        rag = RagAnswerer(retriever)
        log_event("rebuild_index_ok", {"corpus_path": corpus_path, "docs": len(docs), "chunks": len(chunks)})
        return {"status": "index rebuilt", "docs": len(docs), "chunks": len(chunks)}
    except HTTPException:
        raise
    except Exception as e:
        log_event("rebuild_index_error", {"error": str(e)})
        raise HTTPException(status_code=500, detail="Index rebuild failed")

@app.post("/ask")
def ask(req: QueryRequest, corpus_path: str = DEFAULT_CORPUS):
    global rag
    _ensure_ready(corpus_path)
    try:
        result = rag.answer(req.question, top_k=max(1, int(req.top_k)), summarize=bool(req.summarize))
        evidence_files = []
        if isinstance(result, dict) and result.get("evidences"):
            seen = set()
            for e in result["evidences"]:
                fn = e.get("file")
                if fn and fn not in seen:
                    seen.add(fn)
                    evidence_files.append(fn)
        log_event("ask", {"question": req.question, "top_k": int(req.top_k),
                          "summarize": bool(req.summarize), "evidence_files": evidence_files})
        return result
    except Exception as e:
        log_event("ask_error", {"question": req.question, "error": str(e)})
        raise HTTPException(status_code=500, detail="Answer generation failed")

