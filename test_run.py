# test_run.py - FIXED VERSION
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

print("🧪 Testing RAG Pipeline...\n")

# Test 1: Document Processing
print("1. Testing Document Processor...")
try:
    from document_processor import DocumentProcessor
    
    # Create a simple text file instead of PDF for testing
    test_text_file = "test_document.txt"
    with open(test_text_file, "w", encoding="utf-8") as f:
        f.write("Test Document for RAG System\n")
        f.write("This is a test text document.\n")
        f.write("Minimum CGPA requirement is 7.5 for placements.\n")
        f.write("Placement process starts in July each year.\n")
    
    # For PDF testing, you need an actual PDF file
    # For now, we'll skip PDF testing
    print("   ⚠️ PDF test skipped (requires actual PDF file)")
    print("   ✅ DocumentProcessor imported successfully")
    
    os.remove(test_text_file)
    
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 2: Chunking
print("\n2. Testing Intelligent Chunker...")
try:
    from intelligent_chunker import IntelligentChunker
    
    chunker = IntelligentChunker(chunk_size=1000, chunk_overlap=200)
    test_pages = [("This is a long document with multiple paragraphs. " * 50, 
                   {"source": "test.pdf", "page": 1})]
    chunks = chunker.chunk_document(test_pages)
    
    if chunks and len(chunks) > 0:
        print(f"   ✅ Success: Created {len(chunks)} chunks")
        print(f"   📊 Average chunk size: {sum(len(c[0]) for c in chunks) / len(chunks):.0f} chars")
    else:
        print("   ❌ Failed: No chunks created")
        
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 3: Embeddings
print("\n3. Testing Embedding Generator...")
try:
    from embeddings import EmbeddingGenerator
    
    embedder = EmbeddingGenerator()
    test_chunks = [("Sample text for embedding testing.", {"chunk_id": 0})]
    embeddings, metadatas = embedder.generate_embeddings(test_chunks)
    
    if embeddings is not None:
        print(f"   ✅ Success: Generated embeddings")
        print(f"   📊 Shape: {embeddings.shape}")
    else:
        print("   ❌ Failed: No embeddings generated")
        
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 4: Vector Store
print("\n4. Testing Vector Store...")
try:
    from vector_store import SimpleVectorStore
    
    store = SimpleVectorStore()
    store.create_collection()
    
    # Add test documents
    import numpy as np
    test_embeddings = np.array([[0.1] * 384, [0.2] * 384])  # 2D array
    test_metadatas = [{"id": 1, "source": "test1.pdf"}, {"id": 2, "source": "test2.pdf"}]
    test_chunks = ["First document", "Second document"]
    
    store.add_documents(test_embeddings, test_metadatas, test_chunks)
    
    # Test search
    query_embedding = np.array([0.15] * 384)
    results = store.similarity_search(query_embedding, k=2)
    
    if results[0] and len(results[0]) > 0:
        print(f"   ✅ Success: Vector store working")
        print(f"   📊 Retrieved {len(results[0])} documents")
    else:
        print("   ❌ Failed: No results returned")
        
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("✅ Testing completed!")
print("\nTo run the app: streamlit run app.py")