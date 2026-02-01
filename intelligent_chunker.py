import re
from typing import List, Tuple, Dict
import nltk
from nltk.tokenize import sent_tokenize

class IntelligentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Download punkt tokenizer if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def chunk_document(self, pages: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """
        Chunk document pages with intelligent boundaries
        """
        all_chunks = []
        
        for page_text, page_metadata in pages:
            # Split by paragraphs first
            paragraphs = re.split(r'\n\s*\n', page_text)
            current_chunk = ""
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # If paragraph is too large, split by sentences
                if len(para) > self.chunk_size:
                    sentences = sent_tokenize(para)
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) + 1 <= self.chunk_size:
                            temp_chunk += sentence + " "
                        else:
                            if temp_chunk.strip():
                                chunk_metadata = page_metadata.copy()
                                chunk_metadata.update({
                                    "chunk_id": len(all_chunks),
                                    "char_start": page_text.find(temp_chunk[:100]) if temp_chunk else 0
                                })
                                all_chunks.append((temp_chunk.strip(), chunk_metadata))
                            temp_chunk = sentence + " "
                    
                    if temp_chunk.strip():
                        if len(current_chunk) + len(temp_chunk) <= self.chunk_size:
                            current_chunk += temp_chunk
                        else:
                            if current_chunk.strip():
                                chunk_metadata = page_metadata.copy()
                                chunk_metadata.update({
                                    "chunk_id": len(all_chunks),
                                    "char_start": page_text.find(current_chunk[:100]) if current_chunk else 0
                                })
                                all_chunks.append((current_chunk.strip(), chunk_metadata))
                            current_chunk = temp_chunk
                else:
                    # Normal paragraph processing
                    if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                        current_chunk += para + "\n\n"
                    else:
                        if current_chunk.strip():
                            chunk_metadata = page_metadata.copy()
                            chunk_metadata.update({
                                "chunk_id": len(all_chunks),
                                "char_start": page_text.find(current_chunk[:100]) if current_chunk else 0
                            })
                            all_chunks.append((current_chunk.strip(), chunk_metadata))
                        current_chunk = para + "\n\n"
            
            # Add final chunk from page
            if current_chunk.strip():
                chunk_metadata = page_metadata.copy()
                chunk_metadata.update({
                    "chunk_id": len(all_chunks),
                    "char_start": page_text.find(current_chunk[:100]) if current_chunk else 0
                })
                all_chunks.append((current_chunk.strip(), chunk_metadata))
        
        return all_chunks