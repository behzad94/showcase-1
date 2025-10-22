# ==========================================================
# File: src/answer/rag.py
# Goal: Take user query → use Retriever → optionally summarize with LLM
# Features:
#   - Retrieve most relevant document chunks
#   - Compute "citation coverage" to measure confidence
#   - Return honest and explainable answers
#   - If coverage is too low, show safe "no reliable answer" message
#   - Summarize answer with local LLM (Ollama) if enabled
# ==========================================================

from typing import List, Tuple, Dict
from src.search.retriever import Retriever
from src.chunk.chunker import Chunk
from src.answer.summarizer import Summarizer


# ----------------------------------------------------------
# Helper function: shorten long text into preview/snippet
# ----------------------------------------------------------
def _mk_snippet(text: str, max_len: int = 320) -> str:
    """Return a short clean snippet of text for display."""
    t = (text or "").strip().replace("\n", " ")
    return (t[:max_len] + "...") if len(t) > max_len else t


# ----------------------------------------------------------
# Helper function: compute score statistics (best, second, avg)
# ----------------------------------------------------------
def _score_stats(scores: List[float]) -> Dict[str, float]:
    """Calculate simple statistics about retrieval scores."""
    if not scores:
        return {"best": 0.0, "second": 0.0, "avg": 0.0}
    s = sorted(scores, reverse=True)
    return {
        "best": s[0],
        "second": (s[1] if len(s) > 1 else 0.0),
        "avg": sum(scores) / len(scores),
    }


# ----------------------------------------------------------
# Main RAG Answerer Class
# ----------------------------------------------------------
class RagAnswerer:
    """Main class that manages retrieval and answer generation."""

    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.summarizer = Summarizer()

    # ------------------------------------------------------
    # Core function: answer a question from local documents
    # ------------------------------------------------------
    def answer(self, query: str, top_k: int = 3, summarize: bool = False) -> Dict:
        """
        Steps:
          1. Retrieve top chunks with retriever
          2. Compute confidence (coverage)
          3. If coverage < threshold → return safe message
          4. Else → Summarize (if enabled) or show snippet
        """
        # --- 1. Retrieve results ---
        results: List[Tuple[Chunk, float]] = self.retriever.search(query, top_k=top_k)

        # --- No matches at all ---
        if not results:
            return {
                "text": "No relevant passages found in the local documents.",
                "evidences": [],
                "needs_clarification": True,
                "clarify_hint": "Please add more keywords or upload more relevant files.",
                "citation_audit": {"coverage": 0.0, "threshold": 0.12, "warning": True},
            }

        # --- 2. Collect evidences & scores ---
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

        # --- 3. Compute statistics and coverage ---
        stats = _score_stats(scores)
        coverage = round(sum(max(0.0, s) for s in scores) / max(1, len(scores)), 3)
        citation_audit = {
            "coverage": coverage,
            "threshold": 0.15,
            "warning": coverage < 0.15,
        }

        # --- 4. Check for low coverage (too weak evidence) ---
        if coverage < 0.15 or stats["best"] < 0.18:
            # Honest response: no reliable answer found
            return {
                "text": "I could not find a reliable answer in your documents. Please try a more specific question.",
                "evidences": [],
                "needs_clarification": True,
                "clarify_hint": "Your query does not match any relevant content. Please use more specific keywords.",
                "citation_audit": citation_audit,
            }

        # --- 5. Compute low-confidence flag ---
        low_conf = (stats["best"] < 0.20) or ((stats["best"] - stats["second"]) < 0.04)

        # --- 6. Summarize or return top snippet ---
        if summarize:
            # Send chunks to local LLM (Ollama) for summarization
            chunks = [ch for ch, _ in results]
            text = self.summarizer.summarize(query, chunks)
        else:
            # Simple mode: return the top snippet with filename
            top_file = results[0][0].filename
            text = f"Based on the document '{top_file}': " + _mk_snippet(results[0][0].text)

        # --- 7. Return final structured response ---
        return {
            "text": text,
            "evidences": evidences,
            "needs_clarification": bool(low_conf),
            "clarify_hint": (
                "Your query seems ambiguous or weakly matched. "
                "Consider adding more context or keywords."
                if low_conf else ""
            ),
            "citation_audit": citation_audit,
        }

