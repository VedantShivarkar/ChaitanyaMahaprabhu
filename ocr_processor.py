import io
import logging
from typing import Optional

import cv2
import numpy as np
import pytesseract

import fitz  # PyMuPDF


logger = logging.getLogger(__name__)


def preprocess_image_for_ocr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def page_to_image(page: fitz.Page) -> Optional[np.ndarray]:
    try:
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        file_bytes = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image
    except Exception:  # noqa: BLE001
        logger.exception("Failed to render page to image for OCR")
        return None


def extract_ocr_from_page(page: fitz.Page) -> str:
    image = page_to_image(page)
    if image is None:
        return ""
    processed = preprocess_image_for_ocr(image)
    text = pytesseract.image_to_string(processed)
    return text or ""
