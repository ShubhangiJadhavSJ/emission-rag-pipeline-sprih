FROM python:3.11-slim

# System deps: tesseract (OCR fallback) + poppler (pdf2image rasterisation).
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# init_db (tables + ground-truth seed) runs in the FastAPI lifespan handler,
# so a plain uvicorn start is enough — no manual migration step.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
