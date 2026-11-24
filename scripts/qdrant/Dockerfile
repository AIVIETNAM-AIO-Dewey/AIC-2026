FROM python:3.12.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY scripts/qdrant/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/qdrant/ingest.py ./ingest.py
COPY scripts/qdrant/validate_branch1_data.py ./validate_branch1_data.py

ENTRYPOINT ["python", "/app/ingest.py"]
