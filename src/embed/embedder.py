# src/embed/embedder.py
# goal: turn chunks into dense vectors using sentence-transformers (MiniLM L6-v2)
# note: we normalize vectors so inner-product ≈ cosine similarity

from typing import List
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np
from src.chunk.chunker import Chunk

# load model once (fast reuse)
_model = SentenceTransformer("all-MiniLM-L6-v2")

@dataclass
class Embedding:
    chunk_id: str          # which chunk this vector belongs to
    vector: np.ndarray     # 1D float array (shape [384])

def embed_chunk(chunk: Chunk) -> Embedding:
    vec = _model.encode(chunk.text, convert_to_numpy=True, normalize_embeddings=True)
    return Embedding(chunk_id=chunk.chunk_id, vector=vec)

def embed_chunks(chunks: List[Chunk]) -> List[Embedding]:
    texts = [c.text for c in chunks]
    ids   = [c.chunk_id for c in chunks]
    vecs = _model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return [Embedding(chunk_id=i, vector=v) for i, v in zip(ids, vecs)]

