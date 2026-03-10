"""Dense vector store using ChromaDB for embedding-based retrieval."""

from __future__ import annotations

import logging
from pathlib import Path

from aris.core.models import Document

logger = logging.getLogger(__name__)


class DenseStore:
    """Vector store backed by ChromaDB for dense retrieval."""

    def __init__(self, persist_dir: str = ".aris/index/dense", collection_name: str = "aris") -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._collection = None
        self._client = None

    def _ensure_collection(self):
        if self._collection is not None:
            return
        import chromadb
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the dense index."""
        self._ensure_collection()
        ids = []
        doc_texts = []
        metadatas = []
        for doc in documents:
            text = doc.content or doc.snippet or doc.title
            if not text:
                continue
            doc_id = doc.url or str(hash(text))
            ids.append(doc_id)
            doc_texts.append(text[:8000])  # ChromaDB limit
            metadatas.append({
                "url": doc.url,
                "title": doc.title,
                "source": doc.source,
                "snippet": doc.snippet[:500] if doc.snippet else "",
            })

        if ids:
            self._collection.upsert(ids=ids, documents=doc_texts, metadatas=metadatas)

    def query(self, query_text: str, n_results: int = 100) -> list[tuple[Document, float]]:
        """Query the dense index. Returns (Document, score) pairs."""
        self._ensure_collection()
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self._collection.count()),
        )

        documents = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0
                score = max(0.0, 1.0 - distance)  # Convert distance to similarity

                doc = Document(
                    url=meta.get("url", doc_id),
                    title=meta.get("title", ""),
                    snippet=meta.get("snippet", ""),
                    content=results["documents"][0][i] if results["documents"] else "",
                    source=meta.get("source", "index"),
                )
                documents.append((doc, score))

        return documents

    @property
    def count(self) -> int:
        self._ensure_collection()
        return self._collection.count()
