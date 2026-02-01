import logging
from typing import List, Dict, Any

from config import SIMILARITY_THRESHOLD, MAX_TOKEN_BUDGET
from vector_store import query_store
from context_manager import estimate_tokens_for_text


logger = logging.getLogger(__name__)


def _normalize_distances_to_similarities(distances: list[float]) -> list[float]:
    """Map distances to a 0-1 similarity scale.

    Chroma typically returns smaller distances for better matches. We
    normalise distances to [0, 1] where 1 is best and 0 worst. This is
    more robust than a naive `1 - dist` when distances are not bounded
    in [0, 1].
    """
    if not distances:
        return []
    d_min = min(distances)
    d_max = max(distances)
    if d_max == d_min:
        return [1.0 for _ in distances]
    sims: list[float] = []
    for d in distances:
        # smaller distance -> higher similarity
        sim = 1.0 - (float(d) - d_min) / (d_max - d_min)
        sims.append(max(0.0, min(1.0, sim)))
    return sims


def dynamic_retrieve(query: str) -> List[Dict[str, Any]]:
    """Dynamic retrieval with threshold and token-budget stopping.

    We intentionally over-query from the vector store, then select a
    subset based on similarity, diversity (doc/page coverage), and
    token budget, without using a fixed top-K externally. If nothing
    passes the similarity threshold, we fall back to the best few
    chunks (still respecting token budget) so the system remains
    responsive for relevant queries.
    """
    raw = query_store(query, n_results=50)
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    if not documents:
        return []

    # Convert distances to similarities in [0, 1]
    sims = _normalize_distances_to_similarities(distances)

    scored = []
    for doc, meta, sim in zip(documents, metadatas, sims):
        scored.append({"text": doc, "metadata": meta, "score": sim})

    # Highest similarity first
    scored.sort(key=lambda x: x["score"], reverse=True)

    selected: List[Dict[str, Any]] = []
    covered_keys = set()
    total_tokens = 0

    for item in scored:
        score = item["score"]
        if score < SIMILARITY_THRESHOLD:
            # Do not break here; later we may still use low-score
            # items as fallback if nothing passes the threshold.
            continue

        meta = item["metadata"] or {}
        key = (meta.get("doc_id"), meta.get("page_number"))

        diversity_gain = 0 if key in covered_keys else 1
        if not selected and diversity_gain == 0:
            diversity_gain = 1

        est_tokens = estimate_tokens_for_text(item["text"])

        if total_tokens + est_tokens > MAX_TOKEN_BUDGET:
            logger.info("Stopping retrieval due to token budget: %d", total_tokens)
            break

        # Prefer new pages/docs unless the score is significantly higher.
        # Make this check a bit softer so we don't drop moderately relevant
        # chunks from the same page too aggressively.
        if diversity_gain == 0 and score < (SIMILARITY_THRESHOLD + 0.05):
            continue

        selected.append(item)
        total_tokens += est_tokens
        covered_keys.add(key)

    # Fallback: if nothing passed the threshold, still return a few
    # best chunks so that the LLM can attempt an answer using the most
    # relevant available context.
    if not selected:
        fallback: List[Dict[str, Any]] = []
        total_tokens = 0
        for item in scored[:8]:  # best few by similarity
            est_tokens = estimate_tokens_for_text(item["text"])
            if total_tokens + est_tokens > MAX_TOKEN_BUDGET:
                break
            fallback.append(item)
            total_tokens += est_tokens
        selected = fallback

    return selected
