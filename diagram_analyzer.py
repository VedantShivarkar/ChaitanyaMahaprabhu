import logging
from typing import List, Dict, Any

import cv2
import numpy as np

import fitz  # PyMuPDF


logger = logging.getLogger(__name__)


def detect_diagrams_on_page(page: fitz.Page) -> List[Dict[str, Any]]:
    """Very simple heuristic diagram detector.

    Returns list of bounding boxes for large non-text regions, for future use.
    For hackathon demo we only log detections and attach to metadata.
    """
    try:
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        diagrams = []
        h, w = img.shape
        min_area = 0.02 * w * h
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw * ch >= min_area:
                diagrams.append({"x": x, "y": y, "w": cw, "h": ch})
        if diagrams:
            logger.info("Detected %d potential diagrams on page", len(diagrams))
        return diagrams
    except Exception:  # noqa: BLE001
        logger.exception("Diagram detection failed")
        return []
