from typing import List, Tuple, Dict, Any

class ContextFiltering:
def init(self, max_context_tokens: int = 4000):
self.max_context_tokens = max_context_tokens
def filter_context(self, retrieved_documents: List[str], retrieved_metadatas: List[Dict[str, Any]], query: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Filter and format the context for the LLM.

    Returns:
        formatted_context: str
        filtered_metadatas: List[Dict[str, Any]] (only for the chunks included)
    """
    filtered_docs = []
    filtered_metadatas = []
    total_tokens = 0

    for doc, metadata in zip(retrieved_documents, retrieved_metadatas):
        # Rough token count
        doc_tokens = len(doc) // 4
        if total_tokens + doc_tokens > self.max_context_tokens:
            break

        # We can add more filtering logic here, e.g., relevance to query
        filtered_docs.append(doc)
        filtered_metadatas.append(metadata)
        total_tokens += doc_tokens

    # Format the context
    formatted_context = ""
    for i, doc in enumerate(filtered_docs):
        formatted_context += f"Document {i+1} (from {filtered_metadatas[i]['source_file']}, page {filtered_metadatas[i]['page_number']}):\n{doc}\n\n"

    return formatted_context.strip(), filtered_metadatas