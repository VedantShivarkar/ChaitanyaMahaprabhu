# quick_setup.py
import os
import subprocess
import sys

def run_command(command):
    try:
        print(f"Running: {command}")
        subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False

def main():
    print("Setting up RAG Document QA System...")
    
    # 1. Create directories
    print("\n1. Creating directories...")
    directories = [
        "data/uploaded_pdfs",
        "data/processed", 
        "vector_db",
        "test_documents",
        "src"
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✓ Created: {dir_path}")
    
    # 2. Install dependencies (skip if already installed)
    print("\n2. Checking dependencies...")
    
    # Check Python version
    print(f"  Python version: {sys.version}")
    
    # Create minimal requirements
    minimal_reqs = """streamlit==1.28.0
pymupdf==1.23.8
numpy==1.24.3
scikit-learn==1.3.0
nltk==3.8.1
python-dotenv==1.0.0
pypdf==3.17.0
faiss-cpu==1.7.4
pandas==2.0.3
tiktoken==0.5.1"""
    
    with open("requirements_minimal.txt", "w") as f:
        f.write(minimal_reqs)
    
    print("  ✓ Created minimal requirements file")
    
    # 3. Download NLTK data
    print("\n3. Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        print("  ✓ Downloaded NLTK punkt data")
    except:
        print("  ⚠️ Could not download NLTK data")
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Copy all Python files to their respective directories")
    print("2. Run: streamlit run app.py")
    print("3. Upload PDFs and start asking questions!")

if __name__ == "__main__":
    main()