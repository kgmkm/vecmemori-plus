"""Regressions for multi-agent MCP memory reliability and isolation.

The semantic tests use deterministic synthetic vectors; they do not claim
to measure a particular neural model's real-world recall.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from vecmemori import FactRetriever, MemoryStore
import vecmemori.retrieval as retrieval


@pytest.fixture
def memory(tmp_path):
    with MemoryStore(tmp_path / "facts.sqlite", require_embeddings=False) as store:
        yield store


@pytest.fixture
def mcp(tmp_path, monkeypatch):
    import vecmemori.mcp_server as server

    config = SimpleNamespace(
        db_path=str(tmp_path / "mcp.sqlite"),
        embedding_model=None,
        fts_only=True,
        fts_weight=1.0,
        embedding_weight=0.0,
        min_trust=0.3,
        default_trust=0.5,
        fact_language=None,
        category_scope=None,
        read_only=False,
    )
    monkeypatch.setattr(server, "CONFIG", config)
    monkeypatch.setattr(server, "_store", None)
    monkeypatch.setattr(server, "_retriever", None)
    yield server, config
    if server._store is not None:
        server._store.close()


def invoke(server, **args):
    return json.loads(server._handle_fact_store(args))


def feedback(server, **args):
    return json.loads(server._handle_fact_feedback(args))


def test_categories_deduplicate_only_within_category(memory):
    a = memory.add_fact("same exact text", category="agent_a")
    b = memory.add_fact("same exact text", category="agent_b")
    assert a != b
    assert memory.add_fact("same exact text", category="agent_a") == a
    assert {r["category"] for r in memory.list_facts(limit=10)} == {
        "agent_a", "agent_b"
    }


def test_migrate_legacy_schema_preserves_records_and_fts(tmp_path):
    path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE facts (
            fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL UNIQUE,
            fts_text TEXT DEFAULT '',
            category TEXT DEFAULT 'general',
            tags TEXT DEFAULT '',
            trust_score REAL DEFAULT 0.5,
            retrieval_count INTEGER DEFAULT 0,
            helpful_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ruri_embedding BLOB
        );
        CREATE TABLE entities (
            entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            entity_type TEXT DEFAULT 'unknown',
            aliases TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE fact_entities (
            fact_id INTEGER REFERENCES facts(fact_id),
            entity_id INTEGER REFERENCES entities(entity_id),
            PRIMARY KEY(fact_id, entity_id)
        );
        INSERT INTO facts (fact_id, content, fts_text, category)
            VALUES (42, 'legacy historical finding', 'legacy historical finding', 'original');
        INSERT INTO entities (entity_id, name) VALUES (9, 'legacy');
        INSERT INTO fact_entities (fact_id, entity_id) VALUES (42, 9);
    """)
    connection.close()
    with MemoryStore(path, require_embeddings=False) as store:
        row = store._conn.execute(
            "SELECT content, category FROM facts WHERE fact_id=42"
        ).fetchone()
        assert tuple(row) == ("legacy historical finding", "original")
        assert store._conn.execute(
            "SELECT COUNT(*) FROM fact_entities WHERE fact_id=42 AND entity_id=9"
        ).fetchone()[0] == 1
        newer = store.add_fact("legacy historical finding", category="separate")
        assert newer != 42
        ids = {r["fact_id"] for r in store.search_facts("historical", min_trust=0)}
        assert ids == {42, newer}
    # Migration must be idempotent after reopening.
    with MemoryStore(path, require_embeddings=False) as store:
        assert store._conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 2


def test_add_rollback_on_entity_failure(memory, monkeypatch):
    def fail(*_args):
        raise RuntimeError("injected entity-link failure")
    monkeypatch.setattr(memory, "_link_fact_entity", fail)
    with pytest.raises(RuntimeError, match="injected"):
        memory.add_fact('"Fabric Entity" historical finding')
    assert memory._conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    assert memory._conn.execute("SELECT count(*) FROM entities").fetchone()[0] == 0
    assert memory._conn.execute("SELECT count(*) FROM facts_fts").fetchone()[0] == 0


def test_update_rollback_on_embedding_failure(memory, monkeypatch):
    fid = memory.add_fact("original immutable description")
    memory._conn.execute(
        "UPDATE facts SET ruri_embedding = ? WHERE fact_id=?",
        (b"old-vector", fid),
    )
    memory._conn.commit()

    def fail(_content):
        raise RuntimeError("embedding model unavailable")
    monkeypatch.setattr(memory, "_encode_doc_blob", fail)
    with pytest.raises(RuntimeError):
        memory.update_fact(fid, content="changed description")
    row = memory._conn.execute(
        "SELECT content, ruri_embedding FROM facts WHERE fact_id=?", (fid,)
    ).fetchone()
    assert tuple(row) == ("original immutable description", b"old-vector")


def test_update_rollback_on_entity_failure(memory, monkeypatch):
    fid = memory.add_fact('"Original Entity" first text')

    def fail(*_args):
        raise RuntimeError("entity link write failed")
    monkeypatch.setattr(memory, "_link_fact_entity", fail)
    with pytest.raises(RuntimeError):
        memory.update_fact(fid, content='"New Entity" changed text')
    assert memory._conn.execute(
        "SELECT content FROM facts WHERE fact_id=?", (fid,)
    ).fetchone()[0] == '"Original Entity" first text'
    assert memory.search_facts("changed") == []
    assert len(memory.search_facts("first")) == 1


def test_fts_only_update_removes_stale_embedding(memory):
    fid = memory.add_fact("old content")
    memory._conn.execute(
        "UPDATE facts SET ruri_embedding = ? WHERE fact_id=?",
        (b"previous-model-vector", fid),
    )
    memory._conn.commit()
    assert memory.update_fact(fid, content="new content") is True
    row = memory._conn.execute(
        "SELECT ruri_embedding,content FROM facts WHERE fact_id=?", (fid,)
    ).fetchone()
    assert row["content"] == "new content"
    assert row["ruri_embedding"] is None


def test_entity_match_does_not_interpret_sql_like_wildcards(memory):
    memory.add_fact('"Northstar Research" historical record')
    memory.add_fact('"Northstar%" separate record')
    names = [
        r[0] for r in memory._conn.execute(
            "SELECT name FROM entities ORDER BY entity_id"
        ).fetchall()
    ]
    assert names == ["Northstar Research", "Northstar%"]


@pytest.mark.parametrize("invalid", [-1, 0, 101, True, 1.5])
def test_api_search_rejects_invalid_limits(memory, invalid):
    memory.add_fact("searchable record")
    with pytest.raises(ValueError, match="limit"):
        FactRetriever(memory, require_embeddings=False, ruri_weight=0.0).search(
            "searchable", limit=invalid
        )


def test_invalid_direct_fts_syntax_raises_instead_of_semantic_fallback(memory):
    memory.add_fact("safe factual record")
    retriever = FactRetriever(memory, require_embeddings=False, ruri_weight=0.0)
    with pytest.raises(sqlite3.OperationalError):
        retriever.search('"unterminated phrase', limit=3)


def test_semantic_search_can_find_oldest_of_230_records(memory):
    # All 230 facts have embeddings. Only the first is semantically related.
    old_id = memory.add_fact("ancient record with unrelated keywords")
    for i in range(229):
        memory.add_fact(f"newer record number {i}")
    old_vec = np.array([1.0, 0.0], dtype=np.float32).tobytes()
    recent_vec = np.array([0.0, 1.0], dtype=np.float32).tobytes()
    memory._conn.execute("UPDATE facts SET ruri_embedding=?", (recent_vec,))
    memory._conn.execute(
        "UPDATE facts SET ruri_embedding=? WHERE fact_id=?", (old_vec, old_id)
    )
    memory._conn.commit()

    retriever = FactRetriever(memory, require_embeddings=False, ruri_weight=0.0)
    retriever.ruri_weight = 1.0
    retriever.fts_weight = 0.0
    retriever._embedding_available = lambda: True

    with patch.object(
        retrieval, "encode_query_cached",
        return_value=np.array([1.0, 0.0], dtype=np.float32),
    ):
        result = retriever.search("semantic search with no shared term", limit=1)
    assert result[0]["fact_id"] == old_id
    assert result[0]["score"] == pytest.approx(0.5)


def test_shared_retriever_concurrent_queries_do_not_mix_embeddings(memory):
    ids = [
        memory.add_fact("first semantic target"),
        memory.add_fact("second semantic target"),
    ]
    for fid, vec in zip(ids, ([1.0, 0.0], [0.0, 1.0])):
        memory._conn.execute(
            "UPDATE facts SET ruri_embedding=? WHERE fact_id=?",
            (np.array(vec, dtype=np.float32).tobytes(), fid),
        )
    memory._conn.commit()

    retriever = FactRetriever(memory, require_embeddings=False, ruri_weight=0.0)
    retriever.ruri_weight = 1.0
    retriever.fts_weight = 0.0
    retriever._embedding_available = lambda: True

    def embed(text):
        return np.array([1., 0.] if text == "alpha" else [0., 1.],
                        dtype=np.float32)

    barrier = threading.Barrier(2)

    def search(word):
        barrier.wait(timeout=5)
        return retriever.search(word, limit=1)[0]["fact_id"]

    with patch.object(retrieval, "encode_query_cached", side_effect=embed):
        with ThreadPoolExecutor(max_workers=2) as pool:
            for _ in range(10):
                jobs = [pool.submit(search, "alpha"), pool.submit(search, "beta")]
                assert [job.result(timeout=5) for job in jobs] == ids


def test_mcp_rejects_unbounded_requests(mcp):
    server, _ = mcp
    for i in range(24):
        invoke(server, action="add", content=f"searchable item {i:02d}")
    for limit in (-1, 0, 51, True):
        assert "error" in invoke(server, action="search",
                                 query="searchable", limit=limit)
    assert invoke(server, action="search", query="searchable", limit=10)["count"] == 10
    assert "error" in invoke(server, action="reason",
                              entities=["x"] * 9, limit=5)
    assert "error" in invoke(server, action="search",
                             query="?" * 10, limit=5)


def test_mcp_scoped_read_write_and_feedback(mcp):
    server, config = mcp
    agent_a = invoke(server, action="add", content="shared secret agent a",
                     category="agent_a")["fact_id"]
    agent_b = invoke(server, action="add", content="shared secret agent b",
                     category="agent_b")["fact_id"]
    config.category_scope = "agent_a"
    listed = invoke(server, action="list")
    assert [f["fact_id"] for f in listed["facts"]] == [agent_a]
    assert invoke(server, action="search", query="shared")["count"] == 1
    assert "error" in invoke(server, action="list", category="agent_b")
    assert "error" in invoke(server, action="add", content="bad cross-write",
                              category="agent_b")
    assert invoke(server, action="update", fact_id=agent_b,
                  content="modified")["updated"] is False
    assert invoke(server, action="remove", fact_id=agent_b)["removed"] is False
    assert "error" in feedback(server, fact_id=agent_b, action="helpful")
    own = invoke(server, action="add", content="scoped addition")
    assert own["status"] == "added"
    assert server._store._conn.execute(
        "SELECT category FROM facts WHERE fact_id=?", (own["fact_id"],)
    ).fetchone()[0] == "agent_a"
    assert invoke(server, action="list")["count"] == 2


def test_mcp_read_only_blocks_all_mutations(mcp):
    server, config = mcp
    fid = invoke(server, action="add", content="existing record")["fact_id"]
    config.read_only = True
    assert "error" in invoke(server, action="add", content="attempted write")
    assert "error" in invoke(server, action="update", fact_id=fid,
                              trust_delta=0.1)
    assert "error" in invoke(server, action="remove", fact_id=fid)
    assert "error" in feedback(server, action="helpful", fact_id=fid)
    assert invoke(server, action="search", query="existing")["count"] == 1


def test_mcp_rejects_arbitrary_trust_promotion(mcp):
    server, _ = mcp
    fid = invoke(server, action="add", content="unverified memory")["fact_id"]
    assert "error" in invoke(server, action="update", fact_id=fid,
                              trust_delta=100.0)
    assert "error" in invoke(server, action="update", fact_id=fid,
                              trust_delta=float("nan"))
    assert invoke(server, action="update", fact_id=fid,
                  trust_delta=0.1)["updated"] is True


def test_feedback_scores_are_heuristic_not_verified_evidence(memory):
    fid = memory.add_fact("unverified synthetic claim")
    for _ in range(11):
        memory.record_feedback(fid, helpful=True)
    score = memory._conn.execute(
        "SELECT trust_score FROM facts WHERE fact_id=?", (fid,)
    ).fetchone()[0]
    assert score == 1.0
