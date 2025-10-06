# src/answer/summarizer.py
# goal: talk to local Ollama (default http://127.0.0.1:11434) with model "phi3:latest"
# if call fails, we switch to a safe dummy text so UI still works.

import os, json, requests
from typing import List
from src.chunk.chunker import Chunk

class Summarizer:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.model = os.getenv("OLLAMA_MODEL", "phi3:latest")
        self.endpoint = f"{self.base_url}/api/chat"
        self.use_llm = True

    def _chat(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful RAG assistant. Answer ONLY from given text."},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        try:
            r = requests.post(self.endpoint, json=payload, headers=headers, timeout=180)
            if r.status_code >= 400:
                # convert error into readable block (so user understands)
                try:
                    err = r.json()
                except Exception:
                    err = {"raw_text": r.text}
                debug = json.dumps({"endpoint": self.endpoint, "status": r.status_code,
                                    "payload": payload, "response": err}, indent=2, ensure_ascii=False)
                raise requests.HTTPError(f"Ollama error {r.status_code}:\n{debug}")
            data = r.json()
            if "message" in data and "content" in data["message"]:
                return data["message"]["content"].strip()
            if "response" in data:
                return data["response"].strip()
            return "(no message content from Ollama)"
        except Exception as e:
            self.use_llm = False
            return f"(LLM call failed)\n{e}"

    def summarize(self, query: str, chunks: List[Chunk]) -> str:
        # we pass only a compact context so prompt stays small
        text_blocks = [ch.text[:400] for ch in chunks]
        context = "\n\n".join(text_blocks)
        prompt = (
            f"Text:\n{context}\n\n"
            f"Question: {query}\n"
            "If answer not in text, say exactly: 'I cannot find it in the documents.'"
        )
        if not self.use_llm:
            return f"(dummy summary) {context[:200]}..."
        return self._chat(prompt)

