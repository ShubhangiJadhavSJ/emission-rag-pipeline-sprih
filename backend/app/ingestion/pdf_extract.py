"""PDF text extraction with an OCR fallback for scanned documents.

Strategy:
  1. Try the native text layer with pdfplumber (fast, accurate for clean PDFs).
  2. If a page yields little/no text (scanned image), rasterise THAT PAGE ONLY
     with pdf2image and OCR it with Tesseract.

Memory note: we deliberately rasterise one page at a time (first_page/last_page)
rather than the whole PDF at once. Converting an entire large, image-heavy
report to images simultaneously can use 1-2 GB of RAM and was OOM-killing the
backend. Per-page conversion keeps only a single page image in memory.

We also keep tables: pdfplumber's extracted tables are flattened into pipe-
separated rows and appended, because emission figures almost always live in a
table and the linear text layer often mangles their column alignment.
"""
import io
import logging

import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes

logger = logging.getLogger(__name__)

# Below this many characters on a page we assume it's scanned and OCR it.
_MIN_CHARS_PER_PAGE = 40
# Cap OCR work so a huge scanned report can't run away on time/memory.
_MAX_OCR_PAGES = 25
_OCR_DPI = 150


def _ocr_single_page(pdf_bytes: bytes, page_number: int) -> str:
    """Rasterise and OCR a single 1-indexed page, holding one image in memory."""
    try:
        images = convert_from_bytes(
            pdf_bytes, dpi=_OCR_DPI, first_page=page_number, last_page=page_number
        )
        if not images:
            return ""
        try:
            return pytesseract.image_to_string(images[0])
        finally:
            images[0].close()
    except Exception as exc:  # pragma: no cover - tesseract/poppler misconfig
        logger.warning("OCR failed for page %d: %s", page_number, exc)
        return ""


def _flatten_tables(page) -> str:
    out = []
    try:
        for table in page.extract_tables() or []:
            for row in table:
                cells = [(c or "").strip() for c in row]
                if any(cells):
                    out.append(" | ".join(cells))
    except Exception:  # pragma: no cover
        pass
    return "\n".join(out)


def extract_text(pdf_bytes: bytes) -> tuple[str, bool]:
    """Return (full_text, used_ocr).

    used_ocr is surfaced into the trace so we can see which inputs needed the
    scanned-document path.
    """
    used_ocr = False
    parts: list[str] = []
    ocr_pages_done = 0

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = _flatten_tables(page)
            combined = "\n".join(p for p in (text, tables) if p).strip()

            if len(combined) < _MIN_CHARS_PER_PAGE and ocr_pages_done < _MAX_OCR_PAGES:
                # Scanned / image page — OCR just this page.
                ocr_text = _ocr_single_page(pdf_bytes, i + 1)
                ocr_pages_done += 1
                if len(ocr_text) > len(combined):
                    combined = ocr_text
                    used_ocr = True

            if combined:
                parts.append(f"[page {i + 1}]\n{combined}")

    return "\n\n".join(parts), used_ocr
