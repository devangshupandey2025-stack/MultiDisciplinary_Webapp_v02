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

# Use /bin/sh -c to properly expand $PORT
CMD ["/bin/sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
