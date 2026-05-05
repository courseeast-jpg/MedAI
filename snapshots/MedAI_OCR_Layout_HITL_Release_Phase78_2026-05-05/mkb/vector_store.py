"""
MedAI v1.1 — ChromaDB Vector Store (Track A)
Tier-aware semantic search. All embeddings local, no external calls.
"""
from pathlib import Path
from typing import List, Optional
from loguru import logger

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_PATH, EMBEDDING_MODEL, DEDUP_SIMILARITY_THRESHOLD
from app.schemas import MKBRecord


class VectorStore:
    def __init__(self, chroma_path: Path = CHROMA_PATH):
        chroma_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="mkb_records",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Loading embedding model (local)...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Vector store ready. Collection size: {self.collection.count()}")

    def add_record(self, record: MKBRecord) -> List[str]:
        """Embed record content and store with metadata."""
        text = self._build_text(record)
        embedding = self.embedder.encode(text).tolist()
        chunk_id = f"chunk_{record.id}"

        self.collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "record_id": record.id,
                "fact_type": record.fact_type,
                "specialty": record.specialty,
                "tier": record.tier,
                "trust_level": str(record.trust_level),
                "source_type": record.source_type,
            }]
        )
        return [chunk_id]

    def semantic_search(
        self,
        query: str,
        n_results: int = 10,
        specialty: Optional[str] = None,
        tier_filter: Optional[str] = None,
        include_hypothesis: bool = True,
    ) -> List[dict]:
        """Search MKB semantically. Returns list of {text, metadata, distance}."""
        query_embedding = self.embedder.encode(query).tolist()

        where_clause = {}
        if specialty and specialty != "general":
            where_clause["specialty"] = specialty
        if tier_filter:
            where_clause["tier"] = tier_filter
        elif not include_hypothesis:
            where_clause["tier"] = "active"

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, max(1, self.collection.count())),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_clause:
            kwargs["where"] = where_clause

        try:
            results = self.collection.query(**kwargs)
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            output.append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "similarity": 1 - dist,
                "record_id": meta.get("record_id"),
            })
        return output

    def check_duplicate(self, content: str, threshold: float = DEDUP_SIMILARITY_THRESHOLD) -> Optional[str]:
        """Returns existing record_id if near-duplicate found, else None."""
        if self.collection.count() == 0:
            return None
        results = self.semantic_search(content, n_results=1)
        if results and results[0]["similarity"] >= threshold:
            return results[0]["record_id"]
        return None

    def delete_record(self, record_id: str):
        chunk_id = f"chunk_{record_id}"
        try:
            self.collection.delete(ids=[chunk_id])
        except Exception as e:
            logger.warning(f"Could not delete chunk {chunk_id}: {e}")

    def count(self) -> int:
        return self.collection.count()

    def _build_text(self, record: MKBRecord) -> str:
        parts = [record.content]
        if record.structured:
            for k, v in record.structured.items():
                if v:
                    parts.append(f"{k}: {v}")
        return " | ".join(str(p) for p in parts if p)
