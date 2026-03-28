#!/bin/bash
# Startup script for Railway deployment

echo "Starting PlantGuard API..."
echo "PORT=${PORT:-8000}"

exec uvicorn backend.app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
