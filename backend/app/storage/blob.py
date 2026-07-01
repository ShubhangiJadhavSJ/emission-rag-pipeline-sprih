"""Local blob storage for raw PDFs.

Files are written under settings.blob_storage_dir with a content-addressed-ish
name (timestamp + original name) to avoid collisions. This is intentionally a
thin abstraction so it could be swapped for S3/MinIO without touching callers.
"""
import os
import re
import uuid

from app.config import settings


def _safe_name(name: str) -> str:
    base = os.path.basename(name)
    return re.sub(r"[^A-Za-z0-9._-]", "_", base)


def save_pdf(content: bytes, original_name: str) -> str:
    """Persist raw PDF bytes; return the absolute blob path."""
    os.makedirs(settings.blob_storage_dir, exist_ok=True)
    unique = uuid.uuid4().hex[:8]
    fname = f"{unique}_{_safe_name(original_name)}"
    path = os.path.join(settings.blob_storage_dir, fname)
    with open(path, "wb") as fh:
        fh.write(content)
    return path


def read_pdf(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()
