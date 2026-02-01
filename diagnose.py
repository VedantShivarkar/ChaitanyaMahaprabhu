# diagnose.py
import sys
import os

print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("\nChecking imports...")

# Check essential imports
libs = [
    "streamlit",
    "chromadb",
    "numpy",
    "pandas",
    "sklearn",
    "nltk",
    "faiss",
    "fitz",
    "tiktoken"
]

for lib in libs:
    try:
        if lib == "fitz":
            import fitz
        else:
            __import__(lib)
        print(f"✅ {lib}")
    except ImportError as e:
        print(f"❌ {lib}: {e}")

# Check if src directory exists
print("\nChecking project structure...")
for path in ["src", "data", "vector_db"]:
    if os.path.exists(path):
        print(f"✅ {path}/")
    else:
        print(f"❌ {path}/ (missing)")

# Check if src files exist
src_files = ["document_processor.py", "vector_store.py", "embeddings.py"]
for file in src_files:
    path = os.path.join("src", file)
    if os.path.exists(path):
        print(f"✅ {path}")
    else:
        print(f"❌ {path} (missing)")

# Try to import VectorStore
print("\nTrying to import VectorStore...")
try:
    sys.path.insert(0, "src")
    from vector_store import VectorStore
    print("✅ VectorStore imported successfully")
    # Test instantiation
    vs = VectorStore()
    print("✅ VectorStore instantiated successfully")
except Exception as e:
    print(f"❌ Failed to import VectorStore: {e}")
    import traceback
 
    traceback.print_exc()