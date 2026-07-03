#!/usr/bin/env bash
# Start the FastAPI app (development)
set -e
export PYTHONUNBUFFERED=1
uvicorn disaster_nlp.server:app --host 0.0.0.0 --port 8000
