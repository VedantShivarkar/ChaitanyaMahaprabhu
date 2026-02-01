from typing import List, Tuple, Dict, Any
import re

class IntelligentChunker:
def init(self, chunk_size: int = 1000, chunk_overlap: int = 200):
self.chunk_size = chunk_size
self.chunk_overlap = chunk_overlap
def split_by_paragraphs(self, text: str) -> List[str]:
    """Split text by paragraphs (double newline)."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]

def split_by_sentences(self, text: str) -> List[str]:
    """Split text by sentences (using simple regex, can be improved)."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def recursive_chunking(self, text: str, current_chunks: List[str] = None) -> List[str]:
    """Recursively split text into chunks of appropriate size."""
    if current_chunks is None:
        current_chunks = []

    # If the text is already within chunk size, add it as a chunk
    if len(text) <= self.chunk_size:
        if text.strip():
            current_chunks.append(text.strip())
        return current_chunks

    # First, try to split by paragraphs
    paragraphs = self.split_by_paragraphs(text)
    if len(paragraphs) > 1:
        # Try to combine paragraphs into chunks
        chunk = ""
        for para in paragraphs:
            if len(chunk) + len(para) + 2 <= self.chunk_size:  # +2 for newlines
                chunk += para + "\n\n"
            else:
                if chunk.strip():
                    current_chunks.append(chunk.strip())
                chunk = para + "\n\n"
        if chunk.strip():
            current_chunks.append(chunk.strip())
        return current_chunks

    # If one paragraph, try to split by sentences
    sentences = self.split_by_sentences(text)
    if len(sentences) > 1:
        chunk = ""
        for sent in sentences:
            if len(chunk) + len(sent) + 1 <= self.chunk_size:
                chunk += sent + " "
            else:
                if chunk.strip():
                    current_chunks.append(chunk.strip())
                chunk = sent + " "
        if chunk.strip():
            current_chunks.append(chunk.strip())
        return current_chunks

    # If it's one long sentence, split by length (with overlap)
    for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
        chunk = text[i:i + self.chunk_size]
        if chunk.strip():
            current_chunks.append(chunk.strip())
    return current_chunks

def chunk_document(self, pages: List[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
    """Chunk the document pages and return chunks with metadata."""
    chunks = []
    for page_text, page_metadata in pages:
        # Recursively chunk the page text
        page_chunks = self.recursive_chunking(page_text)
        for chunk_text in page_chunks:
            # Create a copy of the page metadata and add chunk-specific info
            chunk_metadata = page_metadata.copy()
            chunk_metadata["chunk_id"] = len(chunks)
            # We can also add the starting and ending character in the page if needed
            chunks.append((chunk_text, chunk_metadata))
    return chunks