# fix_all_errors.py
import os
import sys

print("Fixing all RAG errors...")

# 1. Create simple_vector_store.py
simple_store_code = '''
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity

class SimpleVectorStore:
    def __init__(self, persist_dir: str = "./vector_db"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        
        self.embeddings = None
        self.metadatas = []
        self.chunks = []
        
    def create_collection(self, collection_name: str = "document_qa"):
        print(f"✓ Simple vector store ready")
        return True
    
    def add_documents(self, embeddings: np.ndarray, metadatas: List[Dict], chunks: List[str]):
        if self.embeddings is None:
            self.embeddings = embeddings.astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])
        
        self.metadatas.extend(metadatas)
        self.chunks.extend(chunks)
        
        print(f"✓ Added {len(chunks)} documents")
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 10):
        if self.embeddings is None or len(self.embeddings) == 0:
            return [], [], []
        
        # Calculate cosine similarities
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return results
        documents = [self.chunks[i] for i in top_indices]
        metadatas = [self.metadatas[i] for i in top_indices]
        scores = [float(similarities[i]) for i in top_indices]
        
        return documents, metadatas, scores
'''

with open("src/simple_vector_store.py", "w") as f:
    f.write(simple_store_code)

# 2. Create a simple test app
test_app_code = '''
import streamlit as st
import numpy as np
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("📚 Simple RAG Document QA")

# Session state
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'metadatas' not in st.session_state:
    st.session_state.metadatas = []

# Sidebar
with st.sidebar:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDFs", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and st.button("Process"):
        all_texts = []
        all_metas = []
        
        for file in uploaded_files:
            # Read PDF
            doc = fitz.open(stream=file.read(), filetype="pdf")
            for page_num in range(len(doc)):
                text = doc[page_num].get_text()
                if text.strip():
                    all_texts.append(text[:1000])  # First 1000 chars
                    all_metas.append({
                        "source": file.name,
                        "page": page_num + 1
                    })
            doc.close()
        
        # Create TF-IDF embeddings
        vectorizer = TfidfVectorizer(max_features=300, stop_words="english")
        embeddings = vectorizer.fit_transform(all_texts).toarray().astype(np.float32)
        
        # Store in session
        st.session_state.embeddings = embeddings
        st.session_state.chunks = all_texts
        st.session_state.metadatas = all_metas
        st.session_state.vectorizer = vectorizer
        
        st.success(f"Processed {len(all_texts)} pages")

# Main area
if st.session_state.embeddings is not None:
    question = st.text_input("Ask a question:")
    
    if question:
        # Get query embedding
        vectorizer = st.session_state.vectorizer
        query_vec = vectorizer.transform([question]).toarray().astype(np.float32)
        
        # Calculate similarities
        sims = cosine_similarity(query_vec, st.session_state.embeddings)[0]
        
        # Get top result
        top_idx = np.argmax(sims)
        answer = st.session_state.chunks[top_idx]
        score = sims[top_idx]
        
        # Display
        st.subheader("Answer")
        if score > 0.1:
            st.write(answer[:500])
            st.info(f"Confidence: {score:.3f}")
            
            # Show source
            with st.expander("Source"):
                meta = st.session_state.metadatas[top_idx]
                st.write(f"Document: {meta['source']}, Page: {meta['page']}")
        else:
            st.warning("No relevant information found.")
else:
    st.info("👈 Upload and process PDFs in the sidebar to begin")

st.markdown("---")
st.caption("Simple RAG Demo • TF-IDF Search")
'''

with open("simple_app.py", "w") as f:
    f.write(test_app_code)

print("✅ Created backup files")
print("\nRun this command:")
print("streamlit run simple_app.py")