# test_rag_pipeline.py
import pytest
from src.document_processor import DocumentProcessor
from src.intelligent_chunker import IntelligentChunker
from src.embeddings import EmbeddingGenerator

def test_document_processing():
    """Test PDF text extraction"""
    processor = DocumentProcessor()
    # Use a test PDF
    test_pdf = "test_documents/sample.pdf"
    pages = processor.extract_text_with_metadata(test_pdf)
    assert len(pages) > 0
    assert "page" in pages[0][1]

def test_chunking():
    """Test intelligent chunking"""
    chunker = IntelligentChunker()
    test_text = "This is a test document. It has multiple sentences. " * 50
    metadata = {"source": "test.pdf", "page": 1}
    chunks = chunker.semantic_chunking(test_text, metadata)
    assert len(chunks) > 0
    assert len(chunks[0][0]) <= 1000  # Within chunk size

def test_embeddings():
    """Test embedding generation"""
    embedder = EmbeddingGenerator()
    test_texts = ["Hello world", "Another document"]
    embeddings, _ = embedder.generate_embeddings(
        [(text, {}) for text in test_texts]
    )
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 dimension

def test_dynamic_retrieval():
    """Test retrieval logic"""
    # Mock test - would need actual vector store
    pass

if __name__ == "__main__":
    print("Running RAG pipeline tests...")
    test_document_processing()
    test_chunking()
    test_embeddings()
    print("✅ All tests passed!")