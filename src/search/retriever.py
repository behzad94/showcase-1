# src/search/retriever.py
# goal: combine dense similarity with a tiny keyword overlap boost, then keep
# candidates close to the best score (margin filter) to avoid brittle picks.

from __future__ import annotations
from typing import List, Tuple
import re, numpy as np
from src.store.vectordb import VectorDB
from src.embed.embedder import _model
from src.chunk.chunker import Chunk

_STOP = {"the","a","an","and","or","to","of","in","on","for","is","are","was","were",
         "be","this","that","it","with","as","by","at","from","about","who","what",
         "when","where","which"}

def _tokens(s: str):
    # lower + alnum tokens so it is cheap and deterministic
    toks = re.findall(r"[A-Za-z0-9]+", s.lower())
    return [t for t in toks if t not in _STOP]

def _keyword_score(query: str, text: str) -> float:
    # simple Jaccard-like fraction based on unique tokens
    q = set(_tokens(query))
    if not q:
        return 0.0
    t = set(_tokens(text))
    return len(q.intersection(t)) / len(q)

class Retriever:
    def __init__(self, vectordb: VectorDB, chunks: List[Chunk],
                 keyword_weight: float = 0.15, margin: float = 0.05):
        self.db = vectordb
        self.keyword_weight = float(keyword_weight)
        self.margin = float(margin)
        # build id → chunk map
        self.chunks_map = {}
        for c in chunks:
            if c and c.chunk_id not in self.chunks_map:
                self.chunks_map[c.chunk_id] = c

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        if not query or not query.strip():
            return []
        qvec: np.ndarray = _model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        raw = self.db.search(qvec, top_k=max(top_k * 4, 10))  # over-fetch a bit
        if not raw:
            return []
        scored = []
        for cid, cos in raw:
            ch = self.chunks_map.get(cid)
            if not ch:
                continue
            kw = _keyword_score(query, ch.text)
            total = float(cos) + self.keyword_weight * float(kw)
            scored.append((cid, cos, kw, total))
        if not scored:
            return []
        scored.sort(key=lambda x: x[3], reverse=True)
        best_total = scored[0][3]
        kept = [(cid, total) for (cid, _cos, _kw, total) in scored if total >= best_total - self.margin]
        out: List[Tuple[Chunk, float]] = []
        for cid, total in kept[:max(1, top_k)]:
            ch = self.chunks_map.get(cid)
            if ch:
                out.append((ch, float(total)))
        return out

