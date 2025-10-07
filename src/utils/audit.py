# src/utils/audit.py
# goal: append JSON lines to logs/audit.jsonl for later reflection

import os, json
from datetime import datetime, timezone

LOG_PATH = os.environ.get("AUDIT_LOG_PATH", "logs/audit.jsonl")

def _ensure_dir():
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)

def log_event(kind: str, payload: dict):
    _ensure_dir()
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind, **payload}
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

