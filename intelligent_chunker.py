import logging
from typing import List, Dict, Any

from config import MAX_CHARS_PER_CHUNK, MIN_CHARS_PER_CHUNK
from memory_manager import ensure_enough_memory


logger = logging.getLogger(__name__)


def split_text_into_chunks(text: str, max_chars_per_chunk: int | None = None) -> List[str]:
    max_len = max_chars_per_chunk or MAX_CHARS_PER_CHUNK
    chunks: List[str] = []
    current = []
    current_len = 0

    for paragraph in text.split("\n\n"):
        para = paragraph.strip()
        if not para:
            continue
        para_len = len(para)

        if para_len > max_len:
            # hard split large paragraph
            for i in range(0, para_len, max_len):
                chunks.append(para[i : i + max_len])
            continue

        if current_len + para_len + 2 <= max_len:
            current.append(para)
            current_len += para_len + 2
        else:
            if current:
                chunks.append("\n\n".join(current))
            current = [para]
            current_len = para_len

    if current:
        chunks.append("\n\n".join(current))

    merged: List[str] = []
    buffer = ""
    for c in chunks:
        if len(buffer) + len(c) + 2 <= max_len:
            if buffer:
                buffer += "\n\n" + c
            else:
                buffer = c
        else:
            if buffer:
                merged.append(buffer)
            buffer = c
    if buffer:
        merged.append(buffer)

    return merged


def build_chunks_from_pages(pages: List[Dict[str, Any]], max_chars_per_chunk: int | None = None) -> List[Dict[str, Any]]:
    ensure_enough_memory()
    all_chunks: List[Dict[str, Any]] = []

    for page in pages:
        text = page["text"]
        base_meta = {
            "doc_id": page["doc_id"],
            "filename": page["filename"],
            "page_number": page["page_number"],
        }

        chunks = split_text_into_chunks(text, max_chars_per_chunk=max_chars_per_chunk)
        char_cursor = 0
        for chunk in chunks:
            idx = text.find(chunk, char_cursor)
            if idx == -1:
                idx = char_cursor
            chunk_meta = dict(base_meta)
            chunk_meta["char_start"] = idx
            chunk_meta["char_end"] = idx + len(chunk)

            all_chunks.append(
                {
                    "text": chunk,
                    "metadata": chunk_meta,
                }
            )
            char_cursor = idx + len(chunk)

    return all_chunks
