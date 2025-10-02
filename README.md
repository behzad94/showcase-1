# Document Information Agent (RAG Showcase)

- Ingest: PDF/TXT/MD → chunk (500/50)
- Embed: sentence-transformers all-MiniLM-L6-v2 (normalized)
- Store: FAISS (IP) + manifest
- Retrieve: hybrid (dense + tiny keyword boost)
- Answer: RAG + optional LLM (Ollama phi3), citations, clarification & citation audit
- API: FastAPI
- UI: Streamlit

## Quickstart
python -m pip install -r requirements.txt
uvicorn src.api.server:app --reload
streamlit run src/ui/app.py
