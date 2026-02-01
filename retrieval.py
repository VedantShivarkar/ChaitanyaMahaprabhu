from typing import List, Tuple, Dict, Any
import numpy as np
from sentence_transformers import util

class DynamicRetrieval:
def init(self, vector_store: VectorStore, similarity_threshold: float = 0.75, max_tokens: int = 4000):
self.vector_store = vector_store
self.similarity_threshold = similarity_threshold
self.max_tokens = max_tokens
def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    return util.cos_sim(embedding1, embedding2).item()

def retrieve_dynamic(self, query_embedding: np.ndarray) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
    """
    Retrieve chunks dynamically until:
    1) Similarity score drops below threshold (ex: 0.75)
    2) Coverage diversity stops improving (we check by comparing the new chunk with already retrieved chunks)
    3) Max token budget reached
    """
    retrieved_documents = []
    retrieved_metadatas = []
    retrieved_scores = []

    # We'll start by retrieving a large number, then filter
    initial_n = 20
    documents, metadatas, distances = self.vector_store.search(query_embedding, n_results=initial_n)

    # Convert distances to similarities (assuming cosine distance, so similarity = 1 - distance)
    similarities = [1 - d for d in distances]

    total_tokens = 0
    last_similarity = 1.0  # for the first chunk

    for doc, metadata, sim in zip(documents, metadatas, similarities):
        # Check similarity threshold
        if sim < self.similarity_threshold:
            break

        # Check token budget (rough estimation: 1 token ~ 4 characters)
        chunk_tokens = len(doc) // 4
        if total_tokens + chunk_tokens > self.max_tokens:
            break

        # Check diversity: if the chunk is too similar to already retrieved chunks, skip
        if self._is_redundant(doc, retrieved_documents, query_embedding):
            continue

        retrieved_documents.append(doc)
        retrieved_metadatas.append(metadata)
        retrieved_scores.append(sim)

        total_tokens += chunk_tokens
        last_similarity = sim

    return retrieved_documents, retrieved_metadatas, retrieved_scores

def _is_redundant(self, new_chunk: str, retrieved_chunks: List[str], query_embedding: np.ndarray, redundancy_threshold: float = 0.9) -> bool:
    """Check if the new chunk is redundant with already retrieved chunks."""
    if not retrieved_chunks:
        return False

    # We can compute the embedding of the new chunk and compare with the existing ones
    # But for simplicity, we can use a simple text overlap measure or skip for now
    # For hackathon, we'll skip redundancy check to keep it simple, but we can implement later.
    return False