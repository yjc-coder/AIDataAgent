# AI Data Agent — single uvicorn process serves the API and the Vue3 page.
FROM python:3.12-slim

# Avoid bytecode + keep stdout unbuffered so logs reach docker logs immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code, knowledge base (RAG source), prebuilt Chroma index, and the
# Vue3 chat page (served by the same process via StaticFiles).
COPY app ./app
COPY frontend ./frontend
COPY data ./data

# Bind to 0.0.0.0 so the container is reachable from outside.
EXPOSE 8065

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8065"]
