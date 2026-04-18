"""
MedAI v1.1 — ChromaDB Initialization Script
Run once: python scripts/init_chroma.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.config import CHROMA_PATH
from mkb.vector_store import VectorStore

def init_chroma():
    print("Initializing ChromaDB and loading embedding model...")
    print("(This downloads ~90MB embedding model on first run)")
    vec = VectorStore(CHROMA_PATH)
    print(f"ChromaDB initialized at {CHROMA_PATH}")
    print(f"Collection size: {vec.count()} records")
    print("ChromaDB init complete.")

if __name__ == "__main__":
    init_chroma()
