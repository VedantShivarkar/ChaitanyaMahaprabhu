from typing import List, Dict, Tuple
import re

class EvidenceMapper:
    def __init__(self):
        pass
    
    def find_evidence_locations(self, evidence_text: str, source_text: str) -> List[Tuple[int, int]]:
        """
        Find exact locations of evidence in source text
        Returns list of (start, end) character positions
        """
        locations = []
        
        # Clean the evidence text for matching
        clean_evidence = self._clean_text(evidence_text)
        clean_source = self._clean_text(source_text)
        
        # Try exact match
        start_pos = clean_source.find(clean_evidence)
        if start_pos != -1:
            locations.append((start_pos, start_pos + len(clean_evidence)))
        else:
            # Try fuzzy matching with words
            evidence_words = clean_evidence.split()
            for i in range(len(clean_source.split()) - len(evidence_words) + 1):
                window_words = clean_source.split()[i:i + len(evidence_words)]
                window_text = ' '.join(window_words)
                
                # Simple similarity check
                if self._calculate_similarity(clean_evidence, window_text) > 0.7:
                    # Find character positions
                    char_start = clean_source.find(window_text)
                    if char_start != -1:
                        char_end = char_start + len(window_text)
                        locations.append((char_start, char_end))
                        break
        
        return locations
    
    def _clean_text(self, text: str) -> str:
        """Clean text for matching"""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text.lower()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def highlight_evidence(self, source_text: str, locations: List[Tuple[int, int]]) -> str:
        """
        Generate highlighted text with evidence marked
        """
        if not locations:
            return source_text
        
        # Sort locations by start position
        locations.sort(key=lambda x: x[0])
        
        # Build highlighted text
        result = ""
        last_end = 0
        
        for start, end in locations:
            # Add text before highlight
            result += source_text[last_end:start]
            # Add highlighted text
            result += f"**{source_text[start:end]}**"
            last_end = end
        
        # Add remaining text
        result += source_text[last_end:]
        
        return result