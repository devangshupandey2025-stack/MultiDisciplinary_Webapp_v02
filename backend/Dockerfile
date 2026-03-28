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

# Make start script executable
RUN chmod +x backend/start.sh

ENV PYTHONPATH=/app

EXPOSE 8000

# Use the startup script
CMD ["bash", "backend/start.sh"]
