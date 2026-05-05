"""
MedAI v1.1 — Database Initialization Script
Run once: python scripts/init_db.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from dotenv import load_dotenv
load_dotenv()

from app.config import DB_PATH, CHROMA_PATH, PDF_STORAGE_PATH, PENDING_QUEUE_PATH
from mkb.sqlite_store import SQLiteStore

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    PDF_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    PENDING_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)

    key = os.getenv("DB_ENCRYPTION_KEY", "default_dev_key")
    store = SQLiteStore(DB_PATH, key)
    counts = store.count_records()
    print(f"SQLite initialized at {DB_PATH}")
    print(f"Records: {counts}")
    print("DB init complete.")

if __name__ == "__main__":
    init_db()
