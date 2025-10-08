#!/usr/bin/env bash
# ==========================================================
# RAG Showcase Runner (run.sh)
# Author: Behzad Moloudi
# Description:
#   Starts everything for the Document Information Agent:
#   - activates venv
#   - checks Ollama server and model
#   - starts FastAPI backend
#   - launches Streamlit UI
# ==========================================================

# Stop on first error
set -e

# === CONFIG ===
API_PORT=8000
UI_PORT=8501
OLLAMA_PORT=11434
OLLAMA_MODEL="phi3:latest"
API_URL="http://127.0.0.1:${API_PORT}"
OLLAMA_URL="http://127.0.0.1:${OLLAMA_PORT}"
APP_PATH="src/ui/app.py"
SERVER_PATH="src/api/server:app"

# === HELPER FUNCTIONS ===
log() { echo -e "\033[1;34m[run.sh]\033[0m $1"; }

# === STEP 1: Activate virtual environment ===
if [ ! -d ".venv" ]; then
  log "No virtual environment found — creating..."
  python3 -m venv .venv
fi
source .venv/bin/activate
log "✅ Virtual environment activated"

# === STEP 2: Install dependencies ===
log "📦 Checking dependencies..."
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt >/dev/null
log "✅ All dependencies installed"

# === STEP 3: Check Ollama ===
if ! curl -s "${OLLAMA_URL}/api/tags" >/dev/null; then
  log "🚨 Ollama not responding on ${OLLAMA_URL}"
  log "Please run: 'ollama serve' in another terminal."
  exit 1
fi

# Verify that model exists
if ! curl -s "${OLLAMA_URL}/api/tags" | grep -q "${OLLAMA_MODEL}"; then
  log "💡 Pulling Ollama model ${OLLAMA_MODEL} ..."
  ollama pull "${OLLAMA_MODEL}"
else
  log "✅ Ollama model ${OLLAMA_MODEL} available"
fi

# === STEP 4: Start backend (FastAPI) ===
log "🚀 Starting FastAPI backend on port ${API_PORT}..."

export PYTHONPATH=$(pwd)
uvicorn src.api.server:app --host 127.0.0.1 --port "${API_PORT}" --reload-dir src &
API_PID=$!

# Wait a few seconds for API to start
log "⏳ Waiting for API to respond..."
for i in {1..10}; do
  if curl -s "${API_URL}/docs" >/dev/null; then
    log "✅ API ready on ${API_URL}"
    break
  fi
  sleep 1
done

# === STEP 5: Launch Streamlit UI ===
log "🌐 Launching Streamlit UI on port ${UI_PORT}..."
streamlit run "${APP_PATH}" --server.port "${UI_PORT}"

# === STEP 6: Clean shutdown ===
log "🛑 Shutting down..."
kill "${API_PID}" >/dev/null 2>&1 || true
log "✅ All processes stopped. Goodbye!"
