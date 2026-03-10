"""Index manager coordinates all index types."""

from __future__ import annotations

import logging

from aris.core.config import ArisConfig
from aris.core.models import Document
from aris.index.dense_store import DenseStore
from aris.index.sparse_store import SparseStore
from aris.index.structured_store import StructuredStore

logger = logging.getLogger(__name__)


class IndexManager:
    """Coordinates dense, sparse, and structured indices."""

    def __init__(self, config: ArisConfig) -> None:
        self._config = config
        self._dense = DenseStore(persist_dir=f"{config.index_dir}/dense")
        self._sparse = SparseStore()
        self._structured = StructuredStore(db_path=f"{config.index_dir}/structured.db")

    @property
    def dense(self) -> DenseStore:
        return self._dense

    @property
    def sparse(self) -> SparseStore:
        return self._sparse

    @property
    def structured(self) -> StructuredStore:
        return self._structured

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to all indices."""
        self._dense.add_documents(documents)
        self._sparse.add_documents(documents)
        self._structured.add_documents(documents)
        logger.info("Indexed %d documents across all stores", len(documents))

    def close(self) -> None:
        self._structured.close()
