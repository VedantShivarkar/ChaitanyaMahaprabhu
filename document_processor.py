import fitz  # PyMuPDF
import os
from typing import List, Tuple, Dict, Any
import hashlib
from datetime import datetime
import tempfile

class DocumentProcessor:
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def extract_text_with_metadata(self, pdf_path: str) -> List[Tuple[str, Dict]]:
        """
        Extract text from PDF with page-level metadata
        """
        doc = fitz.open(pdf_path)
        pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Extract metadata
            metadata = {
                "source": os.path.basename(pdf_path),
                "page": page_num + 1,
                "total_pages": len(doc),
                "doc_hash": hashlib.md5(pdf_path.encode()).hexdigest()[:8],
                "extraction_time": datetime.now().isoformat()
            }
            
            if text.strip():
                pages.append((text, metadata))
        
        doc.close()
        return pages
    
    def validate_pdf(self, file_path: str) -> bool:
        """Check if PDF is readable and not corrupted"""
        try:
            with fitz.open(file_path) as doc:
                return len(doc) > 0
        except:
            return False
    
    def process_uploaded_files(self, uploaded_files) -> List[Tuple[str, Dict]]:
        """Process multiple uploaded files"""
        all_pages = []
        
        for uploaded_file in uploaded_files:
            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name
            
            if self.validate_pdf(temp_path):
                pages = self.extract_text_with_metadata(temp_path)
                all_pages.extend(pages)
                
            # Clean up
            os.unlink(temp_path)
        
        return all_pages