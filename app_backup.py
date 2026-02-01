# Main Streamlit App
# app.py - MAIN APPLICATION
import streamlit as st
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from document_processor import DocumentProcessor
from intelligent_chunker import IntelligentChunker
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from dynamic_retriever import DynamicRetriever
from context_manager import ContextManager
from llm_handler import LLMHandler
from evidence_mapper import EvidenceMapper
from confidence_scorer import ConfidenceScorer

# Initialize session state
def init_session_state():
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []

# Page config
st.set_page_config(
    page_title="RAG Document QA Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .confidence-high {
        color: #10B981;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F59E0B;
        font-weight: bold;
    }
    .confidence-low {
        color: #EF4444;
        font-weight: bold;
    }
    .evidence-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .source-highlight {
        background-color: #FEF3C7;
        padding: 2px 4px;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize
init_session_state()

# Sidebar
with st.sidebar:
    st.markdown("## 📚 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multiple PDFs for querying"
    )
    
    if uploaded_files:
        if st.button("🔄 Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                try:
                    # Process documents
                    processor = DocumentProcessor()
                    pages = processor.process_uploaded_files(uploaded_files)
                    
                    # Chunk documents
                    chunker = IntelligentChunker(chunk_size=1000, chunk_overlap=200)
                    all_chunks = []
                    for text, metadata in pages:
                        chunks = chunker.semantic_chunking(text, metadata)
                        all_chunks.extend(chunks)
                    
                    # Generate embeddings
                    embedding_gen = EmbeddingGenerator()
                    embeddings, metadatas = embedding_gen.generate_embeddings(all_chunks)
                    
                    # Create vector store
                    vector_store = VectorStore(store_type="chroma")
                    vector_store.create_collection("document_qa")
                    
                    # Add to vector store
                    chunk_texts = [chunk[0] for chunk in all_chunks]
                    vector_store.add_documents(embeddings, metadatas, chunk_texts)
                    
                    # Update session state
                    st.session_state.vector_store = vector_store
                    st.session_state.documents_processed = True
                    st.session_state.chunks = all_chunks
                    
                    st.success(f"✅ Processed {len(pages)} pages into {len(all_chunks)} chunks")
                    
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
    
    # Database controls
    st.markdown("---")
    st.markdown("## ⚙️ Database Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Database"):
            if st.session_state.vector_store:
                st.session_state.vector_store = None
                st.session_state.documents_processed = False
                st.session_state.chunks = []
                st.success("Database cleared!")
    
    with col2:
        if st.button("📊 Show Stats"):
            if st.session_state.documents_processed:
                st.info(f"""
                **Database Statistics:**
                - Chunks: {len(st.session_state.chunks)}
                - Vector Store: Active
                - Last Processed: Now
                """)
            else:
                st.warning("No documents processed yet")
    
    # Demo documents
    st.markdown("---")
    st.markdown("## 🧪 Demo Documents")
    
    if st.button("Load Demo Documents"):
        # Load pre-processed demo documents
        st.info("Demo feature - would load pre-processed documents")
        st.session_state.documents_processed = True  # Simulate for demo

# Main content
st.markdown('<h1 class="main-header">🔍 RAG Document QA Assistant</h1>', unsafe_allow_html=True)
st.markdown("""
*Upload PDF documents and ask questions with AI-powered semantic search.*  
**Features:** Dynamic retrieval • Source evidence • Confidence scoring • Hallucination prevention
""")

# Check if documents are processed
if not st.session_state.documents_processed:
    st.warning("""
    ⚠️ **No documents processed yet!**
    
    1. Upload PDFs in the sidebar
    2. Click 'Process Documents'
    3. Start asking questions
    """)
    
    # Demo instructions
    with st.expander("🎯 Quick Demo Instructions"):
        st.markdown("""
        **For hackathon demo:**
        1. Prepare these test documents:
           - College policy PDF (e.g., placement rules)
           - Research paper or article
           - Hostel rules PDF
           - Course syllabus
        
        2. Ask questions like:
           - "What is the minimum CGPA for placements?"
           - "What are the hostel visiting hours?"
           - "Explain the key findings of the research"
        
        3. Show features:
           - Dynamic retrieval (not fixed top-K)
           - Source highlighting
           - Confidence scores
           - Refusal when answer not found
        """)
    
    st.stop()

# Question input
st.markdown("## ❓ Ask a Question")
question = st.text_input(
    "Enter your question about the documents:",
    placeholder="e.g., What are the eligibility criteria for campus placements?",
    help="Ask specific questions based on the uploaded documents"
)

if question and st.session_state.vector_store:
    with st.spinner("🔍 Searching documents..."):
        try:
            # Initialize components
            embedding_gen = EmbeddingGenerator()
            retriever = DynamicRetriever(st.session_state.vector_store)
            context_manager = ContextManager()
            llm_handler = LLMHandler(use_openai=False)  # Change to True for OpenAI
            evidence_mapper = EvidenceMapper()
            confidence_scorer = ConfidenceScorer()
            
            # Generate query embedding
            query_embedding = embedding_gen.model.encode(
                [question], 
                convert_to_numpy=True
            )[0]
            
            # Dynamic retrieval
            documents, metadatas, scores = retriever.retrieve_dynamic(
                query_embedding, question
            )
            
            # Filter and format context
            context, selected_metas = context_manager.filter_and_rank_context(
                documents, metadatas, scores, question
            )
            
            # Generate answer
            llm_response = llm_handler.generate_answer(context, question)
            
            # Calculate confidence
            confidence = confidence_scorer.calculate_confidence(
                similarity_scores=scores,
                evidence_found=bool(llm_response["evidence"]),
                context_length=len(context),
                answer_length=len(llm_response["answer"]),
                llm_confidence=llm_response["confidence"]
            )
            
            # Map evidence to source
            evidence_details = []
            for evidence in llm_response["evidence"]:
                # Find source document
                source_doc = next(
                    (doc for doc, meta in zip(documents, metadatas) 
                     if meta.get("source") == evidence["document"]),
                    ""
                )
                
                evidence_detail = evidence_mapper.locate_evidence_in_source(
                    evidence["quote"],
                    source_doc,
                    {"source": evidence["document"], "page": evidence["page"]}
                )
                evidence_details.append(evidence_detail)
            
            # Store in history
            st.session_state.qa_history.append({
                "question": question,
                "answer": llm_response["answer"],
                "confidence": confidence,
                "evidence": evidence_details,
                "timestamp": "now"
            })
            
            # Display results
            st.markdown("---")
            st.markdown("## 💡 Answer")
            
            # Confidence badge
            confidence_color = {
                "High": "confidence-high",
                "Medium": "confidence-medium",
                "Low": "confidence-low"
            }.get(confidence["level"], "")
            
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                <span class="{confidence_color}">Confidence: {confidence["level"]} ({confidence["score"]})</span>
                <progress value="{confidence['score']}" max="1" style="flex-grow: 1;"></progress>
            </div>
            """, unsafe_allow_html=True)
            
            # Answer
            if llm_response["answer"] == "Answer not found in provided documents.":
                st.warning("⚠️ " + llm_response["answer"])
            else:
                st.success(llm_response["answer"])
            
            # Evidence section
            if llm_response["evidence"]:
                st.markdown("## 📑 Source Evidence")
                
                for i, (evidence, detail) in enumerate(zip(llm_response["evidence"], evidence_details)):
                    with st.expander(f"Evidence {i+1}: {evidence['document']} - Page {evidence['page']}"):
                        if detail["found"]:
                            st.markdown(f"**Exact quote:**")
                            st.markdown(f"> {evidence['quote']}")
                            
                            st.markdown(f"**Location in document:**")
                            highlighted = evidence_mapper.highlight_evidence_ui(detail)
                            st.markdown(highlighted)
                        else:
                            st.info("Evidence paraphrased - not found verbatim in document")
            
            # Retrieved chunks (for transparency)
            with st.expander("🔍 View Retrieved Context"):
                st.markdown(f"**Retrieved {len(documents)} chunks:**")
                for i, (doc, meta, score) in enumerate(zip(documents, metadatas, scores)):
                    st.markdown(f"**Chunk {i+1}** (Score: {score:.3f})")
                    st.markdown(f"*Source: {meta.get('source', 'Unknown')}, Page: {meta.get('page', 'N/A')}*")
                    st.text(doc[:200] + "..." if len(doc) > 200 else doc)
                    st.markdown("---")
            
            # Confidence breakdown
            with st.expander("📊 Confidence Breakdown"):
                components = confidence["components"]
                st.markdown(f"**Explanation:** {confidence['explanation']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Similarity", f"{components['similarity']:.2f}")
                with col2:
                    st.metric("Evidence Match", f"{components['evidence_match']:.2f}")
                with col3:
                    st.metric("LLM Confidence", f"{components['llm_confidence']:.2f}")
                
                st.caption("Confidence score: weighted combination of retrieval quality and LLM confidence")
            
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            st.exception(e)

# QA History
if st.session_state.qa_history:
    st.markdown("---")
    st.markdown("## 📜 Question History")
    
    for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):  # Last 5
        with st.expander(f"Q: {qa['question'][:50]}..."):
            st.markdown(f"**A:** {qa['answer']}")
            st.caption(f"Confidence: {qa['confidence']['level']} ({qa['confidence']['score']})")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; padding: 1rem;">
    <small>Built with Streamlit • FAISS/ChromaDB • Sentence Transformers</small><br>
    <small>Dynamic RAG with Hallucination Prevention</small>
</div>
""", unsafe_allow_html=True)