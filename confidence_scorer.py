from typing import List, Dict
import numpy as np

class ConfidenceScorer:
    def __init__(self):
        self.weights = {
            "similarity_score": 0.4,
            "evidence_match": 0.3,
            "context_coverage": 0.2,
            "answer_specificity": 0.1
        }
    
    def calculate_confidence(self, 
                           similarity_scores: List[float],
                           evidence_found: bool,
                           context_length: int,
                           answer_length: int,
                           llm_confidence: str) -> Dict:
        """
        Calculate comprehensive confidence score
        """
        # Normalize similarity scores
        if similarity_scores:
            max_similarity = np.max(similarity_scores)
            avg_similarity = np.mean(similarity_scores)
        else:
            max_similarity = 0
            avg_similarity = 0
        
        # Evidence match score
        evidence_score = 1.0 if evidence_found else 0.0
        
        # Context coverage score
        context_score = min(context_length / 1000, 1.0)
        
        # Answer specificity score
        specificity_score = min(answer_length / 50, 1.0)
        
        # LLM confidence mapping
        llm_score_map = {"High": 0.9, "Medium": 0.6, "Low": 0.3}
        llm_score = llm_score_map.get(llm_confidence, 0.5)
        
        # Calculate weighted score
        weighted_score = (
            self.weights["similarity_score"] * max_similarity +
            self.weights["evidence_match"] * evidence_score +
            self.weights["context_coverage"] * context_score +
            self.weights["answer_specificity"] * specificity_score
        )
        
        # Adjust with LLM confidence
        final_score = (weighted_score * 0.7) + (llm_score * 0.3)
        
        # Determine confidence level
        if final_score >= 0.7:
            confidence_level = "High"
            color = "green"
        elif final_score >= 0.4:
            confidence_level = "Medium"
            color = "orange"
        else:
            confidence_level = "Low"
            color = "red"
        
        return {
            "score": round(final_score, 2),
            "level": confidence_level,
            "color": color,
            "components": {
                "similarity": round(max_similarity, 2),
                "evidence_match": evidence_score,
                "context_coverage": round(context_score, 2),
                "specificity": round(specificity_score, 2),
                "llm_confidence": llm_score
            },
            "explanation": self._generate_explanation(
                confidence_level, 
                max_similarity, 
                evidence_found
            )
        }
    
    def _generate_explanation(self, level: str, similarity: float, evidence_found: bool) -> str:
        """Generate human-readable confidence explanation"""
        if not evidence_found:
            return "Low confidence: No direct evidence found in documents."
        
        if level == "High":
            return f"High confidence: Strong semantic match found (similarity: {similarity:.2f}) with direct evidence."
        elif level == "Medium":
            return f"Medium confidence: Relevant information found but may need verification."
        else:
            return f"Low confidence: Limited evidence found. Answer based on partial information."