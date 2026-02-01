import logging
from typing import List, Dict, Any

import tiktoken

from config import MAX_TOKEN_BUDGET


logger = logging.getLogger(__name__)


def get_token_encoder():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001
        logger.warning("Falling back to simple char-based token estimator")
        return None


_ENCODER = get_token_encoder()


def estimate_tokens_for_text(text: str) -> int:
    if _ENCODER is None:
        return max(1, len(text) // 4)
    return len(_ENCODER.encode(text))


def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    pieces = []
    total_tokens = 0
    for ch in chunks:
        meta = ch.get("metadata", {})
        header = f"[Source: {meta.get('filename','unknown')} | Page {meta.get('page_number','?')}]"
        body = ch["text"].strip()
        snippet = header + "\n" + body + "\n\n"
        tks = estimate_tokens_for_text(snippet)
        if total_tokens + tks > MAX_TOKEN_BUDGET:
            break
        pieces.append(snippet)
        total_tokens += tks
    return "".join(pieces)
