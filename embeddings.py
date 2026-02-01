# src/embeddings.py - FIXED FOR WINDOWS
import numpy as np
from typing import List, Tuple, Dict
import os
import warnings

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_torch: bool = False):
        """
        Initialize with fallback options for Windows
        """
        self.model_name = model_name
        self.model = None
        self.dimension = 384
        self.use_torch = use_torch
        
        # Try to load sentence-transformers
        if use_torch:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
                print(f"✓ Loaded sentence-transformers: {model_name}")
            except Exception as e:
                print(f"⚠️ Could not load sentence-transformers: {e}")
                print("   Falling back to TF-IDF...")
                self._init_tfidf()
        else:
            # Use TF-IDF (no torch dependency)
            self._init_tfidf()
    
    def _init_tfidf(self):
        """Initialize TF-IDF vectorizer"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.model = TfidfVectorizer(max_features=384, stop_words='english')
            print("✓ Using TF-IDF embeddings (CPU-friendly)")
        except ImportError:
            print("⚠️ scikit-learn not available, using random embeddings")
            self.model = None
    
    def generate_embeddings(self, chunks: List[Tuple[str, Dict]]) -> Tuple[np.ndarray, List[Dict]]:
        """Generate embeddings for text chunks"""
        texts = [chunk[0] for chunk in chunks]
        metadatas = [chunk[1] for chunk in chunks]
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        if hasattr(self.model, 'encode'):
            # SentenceTransformers
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        elif hasattr(self.model, 'fit_transform'):
            # TF-IDF
            embeddings = self.model.fit_transform(texts).toarray()
            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings = embeddings / norms
        else:
            # Random embeddings (for demo)
            np.random.seed(42)
            embeddings = np.random.randn(len(texts), self.dimension)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Ensure correct dtype for FAISS
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        print(f"✓ Generated embeddings: {embeddings.shape}")
        
        return embeddings, metadatas
    
    def embed_query(self, text: str) -> np.ndarray:
        """Generate embedding for a single query"""
        if hasattr(self.model, 'encode'):
            embedding = self.model.encode([text], convert_to_numpy=True)[0]
        elif hasattr(self.model, 'transform'):
            embedding = self.model.transform([text]).toarray()[0]
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        else:
            # Random embedding
            np.random.seed(hash(text) % 10000)
            embedding = np.random.randn(self.dimension)
            embedding = embedding / np.linalg.norm(embedding)
        
        return np.ascontiguousarray(embedding, dtype=np.float32)