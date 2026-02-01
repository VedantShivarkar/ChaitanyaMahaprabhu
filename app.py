# app.py - COMPLETE CORRECTED VERSION
import streamlit as st
import os
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Set page config first
st.set_page_config(
    page_title="RAG Document QA Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .confidence-high { color: #10B981; font-weight: bold; }
    .confidence-medium { color: #F59E0B; font-weight: bold; }
    .confidence-low { color: #EF4444; font-weight: bold; }
    .evidence-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

# Sidebar
with st.sidebar:
    st.markdown("# 📚 RAG Document QA")
    st.markdown("---")
    
    # Document upload
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    # Process button
    if st.button("🔄 Process Documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload PDF files first")
        else:
            with st.spinner("Processing documents..."):
                try:
                    # Import with error handling
                    try:
                        from document_processor import DocumentProcessor
                        from intelligent_chunker import IntelligentChunker
                        from embeddings import EmbeddingGenerator
                        from vector_store import SimpleVectorStore
                    except ImportError as e:
                        st.error(f"Import error: {e}")
                        st.stop()
                    
                    # Process each file
                    processor = DocumentProcessor()
                    all_pages = []
                    
                    for uploaded_file in uploaded_files:
                        # Save temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            temp_path = tmp_file.name
                        
                        try:
                            if processor.validate_pdf(temp_path):
                                pages = processor.extract_text_with_metadata(temp_path)
                                all_pages.extend(pages)
                                st.sidebar.success(f"✓ {uploaded_file.name}: {len(pages)} pages")
                        finally:
                            # Clean up temp file
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                    
                    if not all_pages:
                        st.error("No text could be extracted from PDFs")
                        st.stop()
                    
                    # Chunk documents
                    chunker = IntelligentChunker(chunk_size=1000, chunk_overlap=200)
                    all_chunks = chunker.chunk_document(all_pages)
                    
                    # Generate embeddings (without torch to avoid DLL issues)
                    embedding_gen = EmbeddingGenerator(use_torch=False)  # Use TF-IDF
                    embeddings, metadatas = embedding_gen.generate_embeddings(all_chunks)
                    st.session_state.embedding_model = embedding_gen
                    
                    # Create vector store
                    vector_store = SimpleVectorStore()
                    vector_store.create_collection()
                    
                    # Add to vector store
                    chunk_texts = [chunk[0] for chunk in all_chunks]
                    vector_store.add_documents(embeddings, metadatas, chunk_texts)
                    
                    # Update session state
                    st.session_state.vector_store = vector_store
                    st.session_state.documents_processed = True
                    
                    st.success(f"✅ Processed {len(all_pages)} pages into {len(all_chunks)} chunks")
                    
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    st.markdown("---")
    
    # Demo mode
    if st.button("🚀 Load Demo Mode", type="secondary", use_container_width=True):
        st.info("Demo mode activated. You can now ask questions.")
        st.session_state.documents_processed = True
        
        # Create a demo vector store with sample data
        try:
            from vector_store import SimpleVectorStore
            from embeddings import EmbeddingGenerator
            import numpy as np
            
            # Create demo embeddings
            embedding_gen = EmbeddingGenerator(use_torch=False)
            st.session_state.embedding_model = embedding_gen
            
            # Create demo store
            demo_store = SimpleVectorStore()
            demo_store.create_collection()
            st.session_state.vector_store = demo_store
        except Exception as e:
            st.error(f"Error setting up demo: {e}")
        
        st.rerun()
    
    st.markdown("---")
    
    # Database controls
    if st.button("🗑️ Clear Database", type="secondary", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.documents_processed = False
        st.session_state.qa_history = []
        st.success("Database cleared!")
        st.rerun()

# Main content
st.markdown('<h1 class="main-header">🔍 RAG Document QA Assistant</h1>', unsafe_allow_html=True)
st.markdown("""
*Upload PDF documents and ask questions with AI-powered semantic search.*  
**Features:** Dynamic retrieval • Source evidence • Confidence scoring • Hallucination prevention
""")

# Check if documents are processed
if not st.session_state.documents_processed:
    st.info("""
    ### 📋 Getting Started:
    1. **Upload PDFs** in the sidebar (multiple files supported)
    2. **Click 'Process Documents'** to extract and index content
    3. **Ask questions** about your documents
    
    ### 🧪 Quick Demo:
    Click **'Load Demo Mode'** in the sidebar to test without uploading files
    """)
    
    st.stop()

# Question input
st.markdown("## ❓ Ask a Question")
question = st.text_input(
    "Enter your question:",
    placeholder="e.g., What are the main requirements mentioned in the document?",
    key="question_input"
)

if question and st.session_state.vector_store:
    with st.spinner("🔍 Searching documents..."):
        try:
            # Import modules
            try:
                from embeddings import EmbeddingGenerator
                from vector_store import SimpleVectorStore
                from llm_handler import LLMHandler
                from confidence_scorer import ConfidenceScorer
            except ImportError as e:
                st.error(f"Import error: {e}")
                st.stop()
            
            # Get embedding model
            if st.session_state.embedding_model:
                embedding_gen = st.session_state.embedding_model
            else:
                embedding_gen = EmbeddingGenerator(use_torch=False)
                st.session_state.embedding_model = embedding_gen
            
            # Generate query embedding
            query_embedding = embedding_gen.embed_query(question)
            
            # Search in vector store
            vector_store = st.session_state.vector_store
            documents, metadatas, scores = vector_store.similarity_search(query_embedding, k=5)
            
            if not documents:
                st.warning("❌ No relevant information found in the documents.")
                st.session_state.qa_history.append({
                    "question": question,
                    "answer": "No relevant information found.",
                    "confidence": "Low",
                    "timestamp": "Now"
                })
                st.stop()
            
            # Prepare context with better formatting
            context = ""
            for i, (doc, meta, score) in enumerate(zip(documents, metadatas, scores)):
                source = meta.get('source', 'Unknown')
                page = meta.get('page', 'N/A')
                context += f"[Source: {source}, Page: {page}, Relevance: {score:.3f}]\n"
                context += f"{doc}\n\n---\n\n"
            
            # Generate answer with improved LLM
            llm_handler = LLMHandler(use_openai=False)
            llm_response = llm_handler.generate_answer(context, question)
            
            # Calculate confidence
            confidence_scorer = ConfidenceScorer()
            confidence = confidence_scorer.calculate_confidence(
                similarity_scores=scores,
                evidence_found=bool(llm_response.get("evidence", [])),
                context_length=len(context),
                answer_length=len(llm_response.get("answer", "")),
                llm_confidence=llm_response.get("confidence", "Medium")
            )
            
            # Display results
            st.markdown("---")
            
            # Create columns for better layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("## 💡 Answer")
            with col2:
                # Confidence badge
                confidence_level = confidence.get("level", "Medium")
                confidence_color = {
                    "High": "🟢",
                    "Medium": "🟡", 
                    "Low": "🔴"
                }.get(confidence_level, "⚪")
                
                st.markdown(f"### {confidence_color} {confidence_level} Confidence")
                st.caption(f"Score: {confidence.get('score', 0):.2f}")
            
            # Answer box with better styling
            answer = llm_response.get("answer", "No answer generated.")
            
            if "Answer not found" in answer or "not found" in answer.lower():
                st.warning(f"**{answer}**")
            elif "I found information" in answer:
                st.info(f"**{answer}**")
            else:
                st.success(f"**{answer}**")
            
            # Show explanation if available
            explanation = confidence.get("explanation", "")
            if explanation:
                with st.expander("📊 Confidence Analysis"):
                    st.markdown(f"**Why this confidence level?**")
                    st.info(explanation)
                    
                    # Show score breakdown
                    components = confidence.get("components", {})
                    if components:
                        st.markdown("**Score Breakdown:**")
                        for key, value in components.items():
                            st.progress(value, text=f"{key}: {value:.2f}")
            
            # Evidence section
            evidence_list = llm_response.get("evidence", [])
            if evidence_list:
                st.markdown("## 📑 Source Evidence")
                
                for i, evidence in enumerate(evidence_list):
                    with st.expander(f"📄 Evidence {i+1}: {evidence.get('document', 'Unknown')}"):
                        col_a, col_b = st.columns([1, 3])
                        with col_a:
                            st.metric("Page", evidence.get('page', 'N/A'))
                        with col_b:
                            st.metric("Source", evidence.get('document', 'Unknown'))
                        
                        st.markdown("**Quoted Text:**")
                        st.markdown(f"> *{evidence.get('quote', 'No quote available')}*")
            
            # Retrieved context with highlighting
            with st.expander("🔍 View Retrieved Context (Top 5)"):
                st.markdown(f"**Retrieved {len(documents)} most relevant chunks:**")
                
                # Find the chunk that best answers the question
                best_chunk_idx = 0
                if len(scores) > 0:
                    best_chunk_idx = scores.index(max(scores))
                
                for i, (doc, meta, score) in enumerate(zip(documents, metadatas, scores)):
                    # Highlight the best chunk
                    if i == best_chunk_idx:
                        st.markdown(f"### 🏆 **Best Match (Score: {score:.3f})**")
                    else:
                        st.markdown(f"### Chunk {i+1} (Score: {score:.3f})")
                    
                    source_display = f"📄 **{meta.get('source', 'Unknown')}** - Page {meta.get('page', 'N/A')}"
                    st.markdown(source_display)
                    
                    # Display chunk with better formatting
                    chunk_display = doc[:500] + "..." if len(doc) > 500 else doc
                    st.text(chunk_display)
                    st.markdown("---")
            
            # Add to history
            st.session_state.qa_history.append({
                "question": question,
                "answer": answer[:200] + "..." if len(answer) > 200 else answer,
                "confidence": confidence_level,
                "timestamp": "Now"
            })
            
        except Exception as e:
            st.error(f"❌ Error processing question: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Question history
if st.session_state.qa_history:
    st.markdown("---")
    st.markdown("## 📜 Recent Questions")
    
    for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):
        with st.expander(f"Q: {qa['question'][:50]}..."):
            st.markdown(f"**A:** {qa['answer']}")
            st.caption(f"Confidence: {qa['confidence']} • {qa['timestamp']}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; padding: 1rem;">
    <small>Built with Streamlit • FAISS • TF-IDF Embeddings</small><br>
    <small>Dynamic RAG with Hallucination Prevention</small>
</div>
""", unsafe_allow_html=True)