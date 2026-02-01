from typing import List, Dict, Tuple

class ContextManager:
    def __init__(self, max_context_length: int = 4000):
        self.max_context_length = max_context_length
    
    def filter_context(self, documents: List[str], 
                      metadatas: List[Dict], 
                      scores: List[float],
                      query: str) -> Tuple[str, List[Dict]]:
        """
        Filter and format context for LLM
        """
        filtered_docs = []
        filtered_metas = []
        total_length = 0
        
        # Combine similar chunks from same source
        source_map = {}
        for doc, meta, score in zip(documents, metadatas, scores):
            source_key = f"{meta.get('source', 'unknown')}_{meta.get('page', 0)}"
            if source_key not in source_map or score > source_map[source_key]['score']:
                source_map[source_key] = {'doc': doc, 'meta': meta, 'score': score}
        
        # Sort by score and select within length limit
        sorted_items = sorted(source_map.values(), key=lambda x: x['score'], reverse=True)
        
        for item in sorted_items:
            doc = item['doc']
            meta = item['meta']
            
            if total_length + len(doc) <= self.max_context_length:
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                total_length += len(doc)
        
        # Format context
        formatted_context = ""
        for i, (doc, meta) in enumerate(zip(filtered_docs, filtered_metas)):
            source_info = f"[Source: {meta.get('source', 'Unknown')}, Page: {meta.get('page', 'N/A')}]"
            formatted_context += f"{source_info}\n{doc}\n\n---\n\n"
        
        return formatted_context.strip(), filtered_metas