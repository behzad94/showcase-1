#!/usr/bin/env bash
set -e
python -m pip install -r requirements.txt
uvicorn src.api.server:app --reload
