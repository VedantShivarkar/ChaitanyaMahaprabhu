# setup_dirs.py
import os

dirs = [
    "data/uploaded_pdfs",
    "data/processed", 
    "vector_db",
    "test_documents",
    "src"
]

for dir_path in dirs:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created: {dir_path}")

print("✓ All directories created successfully!")
