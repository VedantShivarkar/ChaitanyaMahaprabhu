# debug_faiss.py
import numpy as np
import faiss

print("Testing FAISS...")

# Create test embeddings
test_embeddings = np.random.randn(10, 384).astype(np.float32)
print(f"Test embeddings shape: {test_embeddings.shape}, dtype: {test_embeddings.dtype}")

# Test normalization
try:
    faiss.normalize_L2(test_embeddings)
    print("✓ FAISS normalize_L2 works")
except Exception as e:
    print(f"❌ FAISS normalize_L2 failed: {e}")

# Test index creation
try:
    index = faiss.IndexFlatIP(384)
    index.add(test_embeddings)
    print("✓ FAISS index creation and addition works")
except Exception as e:
    print(f"❌ FAISS index failed: {e}")

print("\n✅ FAISS test complete")