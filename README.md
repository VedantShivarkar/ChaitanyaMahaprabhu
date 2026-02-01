# Document QA RAG Assistant

A hackathon-ready Retrieval Augmented Generation (RAG) system for answering questions over college PDFs (policies, placement brochures, research papers, hostel rules) with **dynamic retrieval**, **hallucination control**, **source highlighting**, and a **Streamlit UI**.

## Features

- **Dynamic retrieval (no fixed K)**
- **Strict hallucination control** with grounded system prompt
- **Source evidence snippets + character ranges** for highlighting
- **Multi-factor confidence scoring** (High/Medium/Low)
- **OCR for scanned PDFs** using Tesseract + OpenCV
- **Memory-aware processing** with timeouts and limits
- **Clean Streamlit UI** with history, progress indicators, and demo mode

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file (optional) based on `.env.example` if you want real OpenAI responses.

## Running the app

```bash
streamlit run app.py
```

## Demo documents

Place the following PDFs inside the `demo_documents/` folder:

- `college_policy.pdf` (placement rules, CGPA requirements)
- `placement_brochure.pdf` (company packages, eligibility)
- `research_paper.pdf` (abstract, methodology, diagrams)
- `hostel_rules.pdf` (timings, regulations)

Then click **"Load Demo Documents"** in the sidebar to pre-index them.

## Judge Explanation Script

> "Unlike basic chatbots, our system uses Retrieval Augmented Generation. It dynamically retrieves only relevant document segments using semantic similarity thresholds. It refuses to answer if content is missing, preventing hallucinations. Every answer is traceable back to original documents."

## Architecture Overview

- `document_processor.py` – PDF loading, text extraction, optional OCR
- `intelligent_chunker.py` – adaptive chunking based on size
- `embeddings.py` – sentence-transformers embedding model
- `vector_store.py` – ChromaDB persistent vector store
- `dynamic_retriever.py` – threshold-based, budget-aware retrieval (no fixed K)
- `context_manager.py` – context assembly and token estimation
- `llm_handler.py` – strict system prompt, OpenAI or mock LLM
- `evidence_mapper.py` – character index mapping for UI highlighting
- `confidence_scorer.py` – multi-factor confidence label & score
- `memory_manager.py` – memory checks and timeout helper
- `ocr_processor.py` – Tesseract OCR with OpenCV preprocessing
- `diagram_analyzer.py` – simple diagram-region detection heuristics
- `app.py` – Streamlit UI wiring everything together

## Notes

- If no `OPENAI_API_KEY` is set, the app uses a **mock LLM** that still obeys the grounding rule and can demo the flow.
- PDF limits and memory thresholds are configured in `config.py` and chosen for hackathon stability.
