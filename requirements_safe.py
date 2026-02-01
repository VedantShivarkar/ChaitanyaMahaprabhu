# In your terminal, run:
pip uninstall chromadb onnxruntime sentence-transformers torch -y

pip install streamlit==1.28.0
pip install pymupdf==1.23.8
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install nltk==3.8.1
pip install python-dotenv==1.0.0
pip install pypdf==3.17.0
pip install faiss-cpu==1.7.4

# Optional: For embeddings, install minimal sentence-transformers
pip install sentence-transformers==2.2.2 --no-deps
pip install transformers==4.35.0
pip install tokenizers==0.14.1
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu