from typing import List, Dict, Any, Tuple


def compute_confidence(
    retrieved_chunks: List[Dict[str, Any]],
    answer: str,
) -> Tuple[str, float]:
    """Return (label, score in [0,1]).

    Heuristic scoring that combines:
    - top similarity
    - average similarity
    - coverage (unique pages/docs)
    - answer specificity (length, presence of numbers)
    """
    if not retrieved_chunks:
        return "Low", 0.0

    scores = [c.get("score", 0.0) for c in retrieved_chunks]
    top = max(scores)
    avg = sum(scores) / len(scores)

    unique_pages = {(c.get("metadata", {}).get("doc_id"), c.get("metadata", {}).get("page_number")) for c in retrieved_chunks}
    coverage_factor = min(1.0, len(unique_pages) / 5.0)

    ans = (answer or "").strip()
    length_factor = min(1.0, len(ans) / 200.0)
    has_number = any(ch.isdigit() for ch in ans)
    number_bonus = 0.1 if has_number else 0.0

    raw = 0.4 * top + 0.2 * avg + 0.2 * coverage_factor + 0.2 * length_factor + number_bonus
    score = max(0.0, min(1.0, raw))

    if score >= 0.75:
        label = "High"
    elif score >= 0.4:
        label = "Medium"
    else:
        label = "Low"
    return label, score
