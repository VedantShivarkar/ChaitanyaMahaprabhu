import os
import logging
from typing import List, Dict, Any

import streamlit as st

from config import (
    APP_TITLE,
    DEMO_DOCS_DIR,
    ENABLE_OCR_DEFAULT,
    MAX_TOTAL_PAGES,
    FAST_MODE_ENABLED_DEFAULT,
    FAST_MODE_MAX_TOTAL_PAGES,
    FAST_MODE_MAX_CHARS_PER_CHUNK,
    MAX_CHARS_PER_CHUNK,
)
from document_processor import extract_documents_from_uploads
from intelligent_chunker import build_chunks_from_pages
from vector_store import add_documents_to_store, reset_collection
from dynamic_retriever import dynamic_retrieve
from context_manager import build_context_block
from llm_handler import answer_question
from evidence_mapper import map_evidence_to_char_positions
from confidence_scorer import compute_confidence
from memory_manager import ensure_enough_memory


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_state():
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    if "db_loaded" not in st.session_state:
        st.session_state.db_loaded = False


def load_demo_documents(enable_ocr: bool = True, fast_mode: bool = False):
    files: List[Any] = []
    if not os.path.isdir(DEMO_DOCS_DIR):
        return
    for name in sorted(os.listdir(DEMO_DOCS_DIR)):
        if name.lower().endswith(".pdf"):
            path = os.path.join(DEMO_DOCS_DIR, name)
            try:
                with open(path, "rb") as f:
                    # Wrap in an object with read/name attributes similar to Streamlit upload
                    class _FileLike:
                        def __init__(self, data, name):
                            self._data = data
                            self.name = name

                        def read(self):
                            return self._data

                    files.append(_FileLike(f.read(), name))
            except Exception:  # noqa: BLE001
                logger.exception("Failed to load demo document %s", path)
    if not files:
        return

    with st.spinner("Pre-loading demo documents..."):
        process_and_store(files, enable_ocr, fast_mode=fast_mode)


def process_and_store(uploads, enable_ocr: bool = True, fast_mode: bool = False):
    ensure_enough_memory()

    progress_bar = st.progress(0.0, text="Parsing PDFs...")

    def _prog(p):
        progress_bar.progress(p, text=f"Parsing PDFs... {int(p*100)}%")

    max_pages = FAST_MODE_MAX_TOTAL_PAGES if fast_mode else MAX_TOTAL_PAGES
    pages = extract_documents_from_uploads(
        uploads,
        enable_ocr=enable_ocr,
        progress_callback=_prog,
        max_total_pages=max_pages,
    )
    progress_bar.progress(1.0, text="PDF parsing complete")

    if not pages:
        st.warning("No text could be extracted from the uploaded PDFs.")
        return

    with st.spinner("Chunking documents..."):
        max_chars = FAST_MODE_MAX_CHARS_PER_CHUNK if fast_mode else MAX_CHARS_PER_CHUNK
        chunks = build_chunks_from_pages(pages, max_chars_per_chunk=max_chars)

    with st.spinner("Creating embeddings and updating vector store..."):
        add_documents_to_store(chunks)

    st.success(f"Indexed {len(chunks)} text chunks from {len(pages)} pages.")
    st.session_state.db_loaded = True


def sidebar_layout():
    st.sidebar.title("Settings & Data")

    st.sidebar.subheader("Upload PDFs")
    uploads = st.sidebar.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    enable_ocr = st.sidebar.checkbox("Enable OCR for scanned PDFs", value=ENABLE_OCR_DEFAULT)
    fast_mode = st.sidebar.checkbox("Fast mode (coarser chunks, fewer pages)", value=FAST_MODE_ENABLED_DEFAULT)

    if st.sidebar.button("Process Uploaded PDFs") and uploads:
        reset_collection()
        process_and_store(uploads, enable_ocr=enable_ocr, fast_mode=fast_mode)

    if st.sidebar.button("Load Demo Documents"):
        reset_collection()
        load_demo_documents(enable_ocr=enable_ocr, fast_mode=fast_mode)

    if st.sidebar.button("Reset Vector Database"):
        reset_collection()
        st.session_state.db_loaded = False
        st.sidebar.success("Vector database reset.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Embedding Stats (approx)")
    if st.session_state.db_loaded:
        st.sidebar.info("Embeddings are available for search.")
    else:
        st.sidebar.warning("No embeddings loaded yet.")

    if fast_mode:
        st.sidebar.caption("Fast mode is ON: using fewer pages and larger chunks for speed.")

    return enable_ocr


def render_confidence(label: str, score: float):
    if label == "High":
        color = "#2ecc71"
    elif label == "Medium":
        color = "#f1c40f"
    else:
        color = "#e74c3c"

    st.markdown(
        f"<div style='padding:0.4rem 0.6rem;border-radius:0.4rem;background:{color}20;border:1px solid {color};display:inline-block;'>"
        f"<b>Confidence:</b> {label} ({score:.2f})"  # noqa: E501
        "</div>",
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_state()

    st.title(APP_TITLE)
    st.caption("Dynamic Retrieval-Augmented Generation with strict hallucination control.")

    enable_ocr = sidebar_layout()

    st.markdown("### Ask a question about your documents")

    examples = [
        "What is the minimum CGPA for placements?",
        "Explain the research methodology",
        "What are hostel visiting hours?",
        "Show me evidence from page 5",
    ]

    cols = st.columns(len(examples))
    example_clicked = None
    for c, ex in zip(cols, examples):
        if c.button(ex):
            example_clicked = ex

    question = st.text_input("Your question", value=example_clicked or "")
    ask = st.button("Answer")

    if ask and question.strip():
        if not st.session_state.db_loaded:
            st.warning("Please upload or load demo documents first.")
        else:
            with st.spinner("Retrieving relevant context..."):
                retrieved = dynamic_retrieve(question)

            if not retrieved:
                st.error("No relevant context found. Try rephrasing the question.")
                answer = "Answer not found in provided documents."
                st.session_state.qa_history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "confidence_label": "Low",
                        "confidence_score": 0.0,
                        "evidence": [],
                    }
                )
            else:
                context = build_context_block(retrieved)
                llm_out = answer_question(context, question)
                answer = llm_out.get("answer", "")

                label, score = compute_confidence(retrieved, answer)
                evidence = map_evidence_to_char_positions(retrieved, question)

                st.markdown("### Answer")
                render_confidence(label, score)
                st.write("")
                st.write(answer)

                with st.expander("View evidence snippets"):
                    for ev in evidence:
                        header = f"**{ev['filename']} - Page {ev['page_number']}**"
                        diag_info = ""
                        if ev.get("diagrams"):
                            diag_info = f"  \\n+Diagrams detected on this page: {len(ev['diagrams'])}"
                        st.markdown(f"{header}{diag_info}\n\n{ev['text']}")

                st.session_state.qa_history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "confidence_label": label,
                        "confidence_score": score,
                        "evidence": evidence,
                    }
                )

    st.markdown("---")
    st.markdown("### QA History")
    if not st.session_state.qa_history:
        st.info("No questions asked yet.")
    else:
        for idx, item in enumerate(reversed(st.session_state.qa_history), start=1):
            with st.expander(f"{idx}. {item['question']}"):
                render_confidence(item["confidence_label"], item["confidence_score"])
                st.write(item["answer"])


if __name__ == "__main__":
    main()
