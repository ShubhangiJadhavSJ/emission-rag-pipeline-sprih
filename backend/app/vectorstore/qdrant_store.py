"""Qdrant wrapper.

One collection holds all chunks; each point carries a `document_id` payload so
retrieval is always scoped to the PDF being extracted (we never want chunks
from one report leaking into another's extraction).
"""
import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import settings
from app.ingestion.embedder import embed_query, embed_texts

logger = logging.getLogger(__name__)


class QdrantStore:
    def __init__(self) -> None:
        # timeout: large reports produce many points; the default client timeout
        # is short and was aborting big upserts with "timed out".
        self.client = QdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port, timeout=120
        )
        self.collection = settings.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(
                    size=settings.embedding_dim, distance=qm.Distance.COSINE
                ),
            )
            # Index on document_id so scoped retrieval is fast.
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="document_id",
                field_schema=qm.PayloadSchemaType.INTEGER,
            )
            logger.info("Created Qdrant collection %s", self.collection)

    def delete_document(self, document_id: int) -> None:
        """Remove any prior chunks for a document (re-ingest is idempotent)."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="document_id",
                            match=qm.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    def index_chunks(self, document_id: int, chunks: list[str]) -> int:
        if not chunks:
            return 0
        vectors = embed_texts(chunks)
        points = [
            qm.PointStruct(
                id=uuid.uuid4().hex,
                vector=vec,
                payload={
                    "document_id": document_id,
                    "chunk_index": idx,
                    "text": chunk,
                },
            )
            for idx, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
        # Upsert in batches so a big document never sends one oversized request.
        batch = 64
        for i in range(0, len(points), batch):
            self.client.upsert(
                collection_name=self.collection, points=points[i : i + batch], wait=True
            )
        return len(points)

    def search(self, document_id: int, query: str, k: int) -> list[dict]:
        vector = embed_query(query)
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=k,
            query_filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="document_id", match=qm.MatchValue(value=document_id)
                    )
                ]
            ),
        )
        return [
            {
                "text": h.payload.get("text", ""),
                "chunk_index": h.payload.get("chunk_index"),
                "score": h.score,
            }
            for h in hits
        ]
