# 🧠 RAG Document QA Assistant

A production-grade **Retrieval-Augmented Generation (RAG) Document Q&A System** built with Python and Streamlit. This system intelligently answers questions from uploaded PDFs by dynamically retrieving relevant context, preventing hallucinations, and providing traceable source evidence—perfect for hackathons and real-world applications.

**Live Demo:** (Add your deployment link here, e.g., Streamlit Cloud, Hugging Face Spaces)

![RAG System Demo](https://img.shields.io/badge/Demo-Available-green) ![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **📁 Multi-PDF Upload** | Process and query across multiple PDF documents simultaneously. |
| **🎯 Dynamic Semantic Retrieval** | Adaptive retrieval based on similarity thresholds, not fixed top-k. |
| **🚫 Hallucination Prevention** | Strict prompting ensures answers are grounded solely in provided context. |
| **🔍 Explainable Source Evidence** | Every answer is linked to exact source text, page, and document. |
| **📊 Confidence Scoring** | Answers include a clear High/Medium/Low confidence score. |
| **💻 CPU-Optimized** | Runs efficiently on CPU using FAISS and TF-IDF/Sentence Transformers. |

## 🏗️ System Architecture
