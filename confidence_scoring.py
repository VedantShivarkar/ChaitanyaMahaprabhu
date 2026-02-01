from typing import List, Dict, Any

class ConfidenceScoring:
def init(self, similarity_scores: List[float]):
self.similarity_scores = similarity_scores
def compute_confidence(self, llm_confidence: str) -> Dict[str, Any]:
    """
    Compute overall confidence based on similarity scores and LLM's confidence.

    We'll convert llm_confidence to a numerical value and combine with average similarity.

    Returns a dictionary with:
        - overall_confidence: High/Medium/Low
        - score: numerical score (0-1)
        - factors: list of factors affecting confidence
    """
    # Convert LLM confidence to numerical
    llm_confidence_map = {"High": 0.9, "Medium": 0.6, "Low": 0.3}
    llm_score = llm_confidence_map.get(llm_confidence, 0.5)

    # Average similarity of retrieved chunks
    avg_similarity = sum(self.similarity_scores) / len(self.similarity_scores) if self.similarity_scores else 0

    # Combine the two scores (weighted average)
    overall_score = 0.7 * avg_similarity + 0.3 * llm_score

    # Convert to High/Medium/Low
    if overall_score >= 0.7:
        overall_confidence = "High"
    elif overall_score >= 0.4:
        overall_confidence = "Medium"
    else:
        overall_confidence = "Low"

    return {
        "overall_confidence": overall_confidence,
        "score": overall_score,
        "factors": [
            f"Average similarity of retrieved chunks: {avg_similarity:.2f}",
            f"LLM confidence: {llm_confidence}"
        ]
    }