import numpy as np
from typing import List, Tuple, Dict

class DynamicRetriever:
    def __init__(self, vector_store, similarity_threshold: float = 0.75, 
                 max_tokens: int = 4000, diversity_threshold: float = 0.9):
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        self.max_tokens = max_tokens
        self.diversity_threshold = diversity_threshold
    
    def retrieve_dynamic(self, query_embedding: np.ndarray, 
                        query_text: str) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Dynamic retrieval with adaptive stopping conditions
        """
        all_documents = []
        all_metadatas = []
        all_scores = []
        
        total_tokens = 0
        max_initial_results = 20
        
        # Get initial batch
        documents, metadatas, scores = self.vector_store.similarity_search(
            query_embedding, k=max_initial_results
        )
        
        for doc, meta, score in zip(documents, metadatas, scores):
            # Stop condition 1: Similarity below threshold
            if score < self.similarity_threshold:
                break
            
            # Stop condition 2: Token budget exceeded
            doc_tokens = len(doc) // 4  # Rough estimate
            if total_tokens + doc_tokens > self.max_tokens:
                break
            
            # Stop condition 3: Redundant content
            if self._is_redundant(doc, all_documents):
                continue
            
            # Add to results
            all_documents.append(doc)
            all_metadatas.append(meta)
            all_scores.append(score)
            total_tokens += doc_tokens
        
        return all_documents, all_metadatas, all_scores
    
    def _is_redundant(self, new_doc: str, existing_docs: List[str]) -> bool:
        """Check if document adds new information"""
        if not existing_docs:
            return False
        
        for existing_doc in existing_docs:
            # Calculate simple word overlap
            new_words = set(new_doc.lower().split())
            existing_words = set(existing_doc.lower().split())
            
            if not new_words or not existing_words:
                continue
            
            overlap = len(new_words.intersection(existing_words)) / len(new_words.union(existing_words))
            if overlap > self.diversity_threshold:
                return True
        
        return False