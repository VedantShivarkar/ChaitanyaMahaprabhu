from typing import List, Dict, Any


def _extract_keywords(text: str) -> List[str]:
    words = [w.strip(".,:;!?()[]{}\"'\n\t").lower() for w in text.split()]
    return [w for w in words if len(w) > 3]


def map_evidence_to_char_positions(
    chunks: List[Dict[str, Any]],
    question: str | None = None,
) -> List[Dict[str, Any]]:
    """Prepare evidence snippets with character ranges and highlight spans.

    Each output item:
    {"filename", "page_number", "char_start", "char_end", "text", "highlights", "diagrams"}
    where "highlights" is a list of {"start", "end", "keyword"} positions
    relative to the original page text.
    """
    keywords = _extract_keywords(question or "")
    mapped = []
    for ch in chunks:
        meta = ch.get("metadata", {})
        base_start = meta.get("char_start")
        text = ch.get("text", "")
        highlights = []
        if base_start is not None and keywords:
            lower_text = text.lower()
            for kw in keywords:
                idx = lower_text.find(kw)
                if idx != -1:
                    highlights.append(
                        {
                            "start": base_start + idx,
                            "end": base_start + idx + len(kw),
                            "keyword": kw,
                        }
                    )

    
        mapped.append(
            {
                "filename": meta.get("filename"),
                "page_number": meta.get("page_number"),
                "char_start": meta.get("char_start"),
                "char_end": meta.get("char_end"),
                "text": text,
                "highlights": highlights,
                "diagrams": meta.get("diagrams"),
            }
        )
    return mapped
