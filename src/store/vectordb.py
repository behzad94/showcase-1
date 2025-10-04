# src/store/vectordb.py
# goal: tiny FAISS wrapper + manifest save/load so our ids survive restarts

from __future__ import annotations
import json, os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import faiss, numpy as np

try:
    from src.embed.embedder import _model as _EMB
    _EMBED_MODEL_NAME = getattr(_EMB, "model_name_or_path", "all-MiniLM-L6-v2")
except Exception:
    _EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

@dataclass
class Manifest:
    dim: int
    id_map: List[str]
    embed_model: str

    @staticmethod
    def load(path: str) -> "Manifest":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return Manifest(dim=int(d["dim"]),
                        id_map=list(d.get("id_map", [])),
                        embed_model=str(d.get("embed_model", _EMBED_MODEL_NAME)))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "id_map": self.id_map, "embed_model": self.embed_model},
                      f, ensure_ascii=False, indent=2)

class VectorDB:
    def __init__(self, dim: int, metric: str = "ip"):
        self.dim = int(dim)
        self.metric = metric.lower().strip()
        self.index = faiss.IndexFlatIP(self.dim) if self.metric != "l2" else faiss.IndexFlatL2(self.dim)
        self.id_map: List[str] = []

    def add_embeddings(self, embeddings) -> None:
        if not embeddings:
            return
        vecs = np.stack([e.vector for e in embeddings]).astype("float32")
        if vecs.shape[1] != self.dim:
            raise ValueError(f"bad dim: got {vecs.shape}, expected (*,{self.dim})")
        self.index.add(vecs)
        self.id_map.extend([e.chunk_id for e in embeddings])

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if query_vec is None:
            return []
        q = np.asarray(query_vec, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self.dim:
            raise ValueError(f"query dim {q.shape[1]} != {self.dim}")
        ntotal = int(self.index.ntotal)
        if ntotal == 0:
            return []
        k = max(1, min(int(top_k), ntotal))
        dists, idxs = self.index.search(q, k)
        out = []
        for i, d in zip(idxs[0], dists[0]):
            if i == -1:
                continue
            sim = (1.0 / (1.0 + float(d))) if self.metric == "l2" else float(d)
            if 0 <= i < len(self.id_map):
                out.append((self.id_map[i], sim))
        return out

    @staticmethod
    def _ensure_dir(path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def save(self, index_path="data/processed/index.faiss",
             manifest_path="data/processed/manifest.json",
             embed_model: Optional[str] = None) -> None:
        embed_model = embed_model or _EMBED_MODEL_NAME
        self._ensure_dir(index_path)
        faiss.write_index(self.index, index_path)
        Manifest(dim=self.dim, id_map=self.id_map, embed_model=embed_model).save(manifest_path)

    @classmethod
    def load(cls, index_path="data/processed/index.faiss",
             manifest_path="data/processed/manifest.json",
             metric="ip") -> "VectorDB":
        if not (os.path.exists(index_path) and os.path.exists(manifest_path)):
            raise FileNotFoundError("index/manifest missing")
        man = Manifest.load(manifest_path)
        db = cls(dim=man.dim, metric=metric)
        db.index = faiss.read_index(index_path)
        db.id_map = list(man.id_map)
        return db

