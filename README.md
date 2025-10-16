# 🧠 Document Information Agent (RAG Showcase)

This project is the **Showcase 1** implementation for the *AI & Interactive Technologies* course.  
It demonstrates a local **Retrieval-Augmented Generation (RAG)** pipeline powered by a local LLM (Ollama phi3).

---

## ⚙️ Architecture Overview

| Stage | Description |
|--------|--------------|
| **Ingest** | Loads PDF / TXT / MD files → detects encoding → extracts text. |
| **Chunk** | Splits into ~500-token chunks with 50 token overlap. |
| **Embed** | Uses `sentence-transformers/all-MiniLM-L6-v2` (normalized vectors). |
| **Store** | Saves vectors in **FAISS (IP)** index + manifest JSON. |
| **Retrieve** | Hybrid dense retrieval + small keyword overlap boost. |
| **Answer** | RAG pipeline with optional **Ollama phi3** summarization, clarifications, and citation-audit logic. |
| **API** | FastAPI backend exposing `/ask` and `/rebuild_index`. |
| **UI** | Streamlit interface with transparency & GDPR notice. |

---

## 🚀 Quick Start

```bash
# 1️⃣ Install dependencies
python -m pip install -r requirements.txt

# 2️⃣ (Optional) create and activate virtual env
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux

# 3️⃣ Start backend API
uvicorn src.api.server:app --reload-dir src

# 4️⃣ Launch Streamlit UI
streamlit run src/ui/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧩 Local LLM (Ollama)

This showcase runs **fully locally** for data-privacy reasons.

```bash
ollama serve
ollama pull phi3:latest
```

The summarizer connects to
`http://127.0.0.1:11434/api/chat` using `phi3:latest`.

---

## 📁 Folder Structure

```
showcase1-rag/
├── corpus/           # user documents (PDF/TXT/MD)
├── data/processed/   # FAISS index + manifest
├── logs/             # JSONL audit logs
├── src/
│   ├── api/server.py
│   ├── ingest/, chunk/, embed/, search/, store/, answer/
│   └── ui/app.py
├── run.sh            # unified launcher script
├── requirements.txt
└── README.md
```

---

## 🧾 Transparency & Ethics

* The system operates **offline**, no external API calls.
* Citations and confidence checks are provided for transparency.
* Users must review outputs before making impactful decisions (GDPR Art. 22).

---

## 🧪 One-Command Run

```bash
./run.sh
```

This script:

1. Activates `.venv`
2. Installs dependencies if missing
3. Checks Ollama + model
4. Starts FastAPI backend
5. Launches Streamlit UI
6. Cleans up on exit

---

## 🧑‍💻 Author

**Behzad Moloudi** — Turku UAS, 2025

```