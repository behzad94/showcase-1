# src/answer/rag.py
# goal: take user query → use Retriever → optionally summarize with LLM
# we also compute simple stats and a "citation coverage" to show transparency.

from typing import List, Tuple, Dict
from src.search.retriever import Retriever
from src.chunk.chunker import Chunk
from src.answer.summarizer import Summarizer

def _mk_snippet(text: str, max_len: int = 320) -> str:
    t = (text or "").strip().replace("\n", " ")
    return (t[:max_len] + "...") if len(t) > max_len else t

def _score_stats(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {"best": 0.0, "second": 0.0, "avg": 0.0}
    s = sorted(scores, reverse=True)
    return {"best": s[0], "second": (s[1] if len(s) > 1 else 0.0), "avg": sum(scores)/len(scores)}

class RagAnswerer:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.summarizer = Summarizer()

    def answer(self, query: str, top_k: int = 3, summarize: bool = False) -> Dict:
        results: List[Tuple[Chunk, float]] = self.retriever.search(query, top_k=top_k)
        if not results:
            return {
                "text": "No relevant passages found in the local documents.",
                "evidences": [],
                "needs_clarification": True,
                "clarify_hint": "Please add more keywords or upload more relevant files.",
                "citation_audit": {"coverage": 0.0, "threshold": 0.12, "warning": True},
            }

        evidences, scores = [], []
        for ch, sc in results:
            scores.append(float(sc))
            evidences.append({
                "file": ch.filename,
                "score": float(sc),
                "snippet": _mk_snippet(ch.text),
                "page_start": getattr(ch, "page_start", None),
                "page_end": getattr(ch, "page_end", None),
            })

        stats = _score_stats(scores)
        coverage = round(sum(max(0.0, s) for s in scores) / max(1, len(scores)), 3)
        low_conf = (stats["best"] < 0.13) or ((stats["best"] - stats["second"]) < 0.03)

        if summarize:
            chunks = [ch for ch, _ in results]
            text = self.summarizer.summarize(query, chunks)
        else:
            top_file = results[0][0].filename
            text = f"Based on the document '{top_file}': " + _mk_snippet(results[0][0].text)

        return {
            "text": text,
            "evidences": evidences,
            "needs_clarification": bool(low_conf),
            "clarify_hint": "Your query seems ambiguous or weakly matched. Consider adding more context or keywords."
                             if low_conf else "",
            "citation_audit": {"coverage": coverage, "threshold": 0.12, "warning": coverage < 0.12},
        }

