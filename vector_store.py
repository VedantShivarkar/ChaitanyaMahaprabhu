# src/vector_store.py - SIMPLE FIXED VERSION
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple

class SimpleVectorStore:
    def __init__(self, persist_dir: str = "./vector_db"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        
        self.embeddings = None
        self.metadatas = []
        self.chunks = []
        
        # Try to load existing data
        self.load_from_disk()
    
    def create_collection(self, collection_name: str = "document_qa"):
        """Initialize the vector store"""
        print("✓ Simple vector store ready")
        return True
    
    def add_documents(self, embeddings: np.ndarray, metadatas: List[Dict], chunks: List[str]):
        """Add documents to vector store"""
        if len(embeddings) == 0:
            print("⚠️ No embeddings to add")
            return
        
        if self.embeddings is None:
            self.embeddings = embeddings.astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])
        
        self.metadatas.extend(metadatas)
        self.chunks.extend(chunks)
        
        # Save to disk
        self._save_to_disk()
        print(f"✓ Added {len(chunks)} documents to store")
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 10):
        """Search for similar documents"""
        if self.embeddings is None or len(self.embeddings) == 0:
            return [], [], []
        
        # Calculate cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k indices
        if len(similarities) < k:
            k = len(similarities)
        
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return results
        documents = [self.chunks[i] for i in top_indices]
        metadatas = [self.metadatas[i] for i in top_indices]
        scores = [float(similarities[i]) for i in top_indices]
        
        return documents, metadatas, scores
    
    def _save_to_disk(self):
        """Save vector store to disk"""
        try:
            data = {
                "embeddings": self.embeddings,
                "metadatas": self.metadatas,
                "chunks": self.chunks
            }
            with open(os.path.join(self.persist_dir, "vector_store.pkl"), 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"⚠️ Could not save to disk: {e}")
    
    def load_from_disk(self):
        """Load vector store from disk"""
        try:
            path = os.path.join(self.persist_dir, "vector_store.pkl")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data.get("embeddings")
                    self.metadatas = data.get("metadatas", [])
                    self.chunks = data.get("chunks", [])
                print(f"✓ Loaded existing store with {len(self.chunks)} chunks")
                return True
        except Exception as e:
            print(f"⚠️ Could not load from disk: {e}")
        return False