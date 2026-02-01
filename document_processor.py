import io
import logging
from typing import List, Dict, Any

import fitz  # PyMuPDF

from config import MAX_PDF_PAGES_PER_FILE, MAX_TOTAL_PAGES
from memory_manager import ensure_enough_memory
from ocr_processor import extract_ocr_from_page
from diagram_analyzer import detect_diagrams_on_page


logger = logging.getLogger(__name__)


def load_pdf_bytes(file_bytes: bytes, filename: str) -> fitz.Document:
    try:
        doc = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")
    except Exception as e:
        logger.exception("Failed to open PDF %s", filename)
        raise ValueError(f"Could not open PDF {filename}: {e}") from e
    if doc.page_count > MAX_PDF_PAGES_PER_FILE:
        raise ValueError(
            f"PDF {filename} has {doc.page_count} pages which exceeds the limit of {MAX_PDF_PAGES_PER_FILE}."
        )
    return doc


def extract_documents_from_uploads(
    uploads: List[Any],
    enable_ocr: bool = True,
    progress_callback=None,
    max_total_pages: int | None = None,
) -> List[Dict[str, Any]]:
    """Return list of page-level dicts with text and metadata.

    Each element: {"doc_id", "filename", "page_number", "text", "char_offset", "metadata"}
    """
    pages: List[Dict[str, Any]] = []
    total_files = len(uploads)
    max_pages = max_total_pages or MAX_TOTAL_PAGES
    total_pages_seen = 0

    for idx, uploaded in enumerate(uploads):
        filename = getattr(uploaded, "name", f"file_{idx}.pdf")
        file_bytes = uploaded.read() if hasattr(uploaded, "read") else uploaded.getvalue()

        ensure_enough_memory()

        if total_pages_seen >= max_pages:
            logger.warning("Global page limit reached (%d); skipping remaining PDFs", max_pages)
            break

        doc = load_pdf_bytes(file_bytes, filename)
        doc_id = f"{filename}"

        for page_index in range(doc.page_count):
            if total_pages_seen >= max_pages:
                logger.info("Reached max total pages (%d); truncating remaining pages", max_pages)
                break
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""

            if enable_ocr and not text.strip():
                try:
                    text = extract_ocr_from_page(page)
                except Exception:
                    logger.exception("OCR failed on %s page %d", filename, page_index + 1)

            if not text.strip():
                continue

            diagrams = detect_diagrams_on_page(page)

            pages.append(
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "page_number": page_index + 1,
                    "text": text,
                    "char_offset": 0,
                    "metadata": {
                        "source": filename,
                        "page": page_index + 1,
                        "diagrams": diagrams,
                    },
                }
            )
            total_pages_seen += 1

        if progress_callback:
            progress_callback((idx + 1) / max(total_files, 1.0))

    return pages
