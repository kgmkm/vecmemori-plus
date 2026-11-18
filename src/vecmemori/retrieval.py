"""Hybrid keyword/neural retrieval for the memory store.

Combines FTS5 full-text search with neural embedding similarity.
"""

from __future__ import annotations

import logging
import math
import heapq
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryStore

logger = logging.getLogger(__name__)

from ._tokenizer import tokenize_query

try:
    import numpy as np
    from ._embedder import encode_query_cached, is_available as embedding_is_available
    _HAS_EMBEDDER_MODULE = True
except ImportError:
    _HAS_EMBEDDER_MODULE = False


class FactRetriever:
    """FTS5 + neural embedding hybrid fact retrieval with trust-weighted scoring."""

    def __init__(
        self,
        store: MemoryStore,
        temporal_decay_half_life: int = 0,  # days, 0 = disabled
        fts_weight: float = 0.40,
        ruri_weight: float = 0.60,
        ruri_keep_alive: int = -1,  # -1=always loaded, 0=unload after search, N=keep N seconds
        require_embeddings: bool = True,
    ):
        self.store = store
        self.half_life = temporal_decay_half_life
        self._ruri_keep_alive = ruri_keep_alive
        self.require_embeddings = require_embeddings

        if ruri_weight > 0 and require_embeddings and not self._embedding_available():
            raise RuntimeError(
                "Embedding model is required for vecmemori retrieval. Install vecmemori "
                "and download/configure a local SentenceTransformer model."
            )

        # Explicit FTS-only mode is available for tests and diagnostics only.
        if ruri_weight > 0 and not require_embeddings and not self._embedding_available():
            fts_weight += ruri_weight
            ruri_weight = 0.0

        self.fts_weight = fts_weight
        self.ruri_weight = ruri_weight

    def search(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
        fts_query: str | None = None,
    ) -> list[dict]:
        """Return the highest-scoring lexical AND semantic hits.

        Hybrid mode scans all stored vectors in bounded batches, not only
        recent facts or FTS matches. The query embedding is local to this
        call, so concurrent requests cannot overwrite each other's state.
        """
        if type(limit) is not int or not (1 <= limit <= 100):
            raise ValueError("limit must be an integer from 1 to 100")
        if not isinstance(query, str):
            raise ValueError("query must be text")
        if not query.strip():
            return []
        if not math.isfinite(min_trust) or not (0 <= min_trust <= 1):
            raise ValueError("min_trust must be finite and between 0 and 1")

        candidates = {
            f["fact_id"]: f for f in self._fts_candidates(
                fts_query if fts_query is not None else query,
                category, min_trust, limit * 3,
            )
        }
        use_embeddings = self.ruri_weight > 0 and self._embedding_available()
        query_emb = self._get_query_embedding(query) if use_embeddings else None

        # Keep only top-k candidates in memory regardless of corpus size.
        heap: list[tuple[float, int, dict]] = []

        def consider(fact: dict, embedding_sim: float) -> None:
            fts_score = fact.pop("fts_rank", 0.0)
            relevance = self.fts_weight * fts_score + self.ruri_weight * embedding_sim
            score = relevance * fact["trust_score"]
            if self.half_life > 0:
                score *= self._temporal_decay(
                    fact.get("updated_at") or fact.get("created_at")
                )
            if not math.isfinite(score) or score <= 0:
                return
            fact["score"] = score
            fact.pop("ruri_embedding", None)
            item = (score, fact["fact_id"], fact)
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)

        if use_embeddings:
            where = ["trust_score >= ?", "ruri_embedding IS NOT NULL"]
            params: list = [min_trust]
            if category is not None:
                where.append("category = ?")
                params.append(category)
            sql = "SELECT * FROM facts WHERE " + " AND ".join(where)

            with self.store._lock:
                cursor = self.store._conn.execute(sql, params)
                while batch := cursor.fetchmany(256):
                    for row in batch:
                        fid = row["fact_id"]
                        fact = candidates.pop(fid, None) or dict(row)
                        raw = row["ruri_embedding"]
                        if not raw or len(raw) % 4:
                            consider(fact, 0.0)
                            continue
                        vec = np.frombuffer(raw, dtype=np.float32)
                        if vec.shape != query_emb.shape:
                            consider(fact, 0.0)
                            continue
                        sim = float(np.dot(query_emb, vec))
                        consider(fact, sim)

        for fact in candidates.values():
            consider(fact, 0.0)

        if self._ruri_keep_alive == 0 and _HAS_EMBEDDER_MODULE:
            from ._embedder import unload_model
            unload_model()
        return [item[2] for item in sorted(heap, reverse=True)]

    def _embedding_available(self) -> bool:
        return _HAS_EMBEDDER_MODULE and embedding_is_available()

    def _get_query_embedding(self, query: str) -> "np.ndarray":
        """Use global cached encoding without storing mutable per-query state."""
        if not _HAS_EMBEDDER_MODULE:
            raise RuntimeError("Embedding dependencies are not installed")
        vec = encode_query_cached(query)
        if vec is None:
            raise RuntimeError("Embedding model failed to encode query")
        return vec

    def _fts_candidates(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
    ) -> list[dict]:
        """Get raw FTS5 candidates from the store.

        Uses the store's database connection directly for FTS5 MATCH
        with rank scoring. Normalizes FTS5 rank to [0, 1] range.
        """
        conn = self.store._conn

        params: list = []
        where_clauses = ["facts_fts MATCH ?"]
        fts_query = tokenize_query(query)
        params.append(fts_query)

        if category:
            where_clauses.append("f.category = ?")
            params.append(category)

        where_clauses.append("f.trust_score >= ?")
        params.append(min_trust)

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT f.*, facts_fts.rank as fts_rank_raw
            FROM facts_fts
            JOIN facts f ON f.fact_id = facts_fts.rowid
            WHERE {where_sql}
            ORDER BY facts_fts.rank
            LIMIT ?
        """
        params.append(limit)

        with self.store._lock:
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return []

        raw_ranks = [abs(row["fts_rank_raw"]) for row in rows]
        max_rank = max(raw_ranks) if raw_ranks else 1.0
        max_rank = max(max_rank, 1e-6)

        results = []
        for row, raw_rank in zip(rows, raw_ranks):
            fact = dict(row)
            fact.pop("fts_rank_raw", None)
            fact["fts_rank"] = raw_rank / max_rank
            results.append(fact)

        return results

    def _temporal_decay(self, timestamp_str: str | None) -> float:
        """Exponential decay: 0.5^(age_days / half_life_days).

        Returns 1.0 if decay is disabled or timestamp is missing.
        """
        if not self.half_life or not timestamp_str:
            return 1.0

        try:
            if isinstance(timestamp_str, str):
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                ts = timestamp_str

            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
            if age_days < 0:
                return 1.0

            return math.pow(0.5, age_days / self.half_life)
        except (ValueError, TypeError):
            return 1.0
