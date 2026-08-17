FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY backend/pyproject.toml ./backend/pyproject.toml
COPY backend/src ./backend/src
RUN pip install --no-cache-dir ./backend
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "aic_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
