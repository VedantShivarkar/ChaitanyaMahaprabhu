import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store_data")
DEMO_DOCS_DIR = os.path.join(BASE_DIR, "demo_documents")

# Retrieval / RAG settings
SIMILARITY_THRESHOLD = 0.6
MAX_TOKEN_BUDGET = 5000
MAX_CHARS_PER_CHUNK = 1000
MIN_CHARS_PER_CHUNK = 400
MAX_PDF_PAGES_PER_FILE = 50
MAX_TOTAL_PAGES = 150

# Fast mode settings (coarser chunks, fewer pages for speed)
FAST_MODE_ENABLED_DEFAULT = False
FAST_MODE_MAX_CHARS_PER_CHUNK = 2000
FAST_MODE_MAX_TOTAL_PAGES = 80

# Memory / safety
MIN_FREE_MEMORY_RATIO = 0.1  # stop if free RAM < 10%
PROCESSING_TIMEOUT_SECONDS = 180

# Embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768

# LLM
SYSTEM_PROMPT = (
    "You are a strict document-based AI assistant. RULES: 1. Use ONLY the provided context. "
    "2. If answer not found, respond exactly: 'Answer not found in provided documents.' "
    "3. Do not assume. 4. Do not hallucinate. 5. Quote source evidence."
)

# OCR / Vision
ENABLE_OCR_DEFAULT = True
ENABLE_VISION_DEFAULT = False

# Streamlit / UI
APP_TITLE = "Document QA RAG Assistant"
