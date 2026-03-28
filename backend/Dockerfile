FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY backend/ ./backend/
COPY ml_pipeline/ ./ml_pipeline/

# Copy model checkpoints if available
COPY checkpoints/ ./checkpoints/

ENV PYTHONPATH=/app

EXPOSE 8000

# Use Python to read PORT env var (guaranteed to work)
CMD ["python", "-c", "import os; import subprocess; port = os.environ.get('PORT', '8000'); subprocess.run(['uvicorn', 'backend.app.main:app', '--host', '0.0.0.0', '--port', port])"]
