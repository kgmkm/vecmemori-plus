"""Embedding-mode test: full hybrid path (FTS5 + ruri-v3) through the MCP handlers.

Skipped automatically when the ruri-v3 model is not present locally.
Download it with: bash scripts/download_model.sh
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO_SRC = Path(__file__).resolve().parent.parent / "src"
DEFAULT_MODEL = Path.home() / ".cache" / "vecmemori" / "models" / "ruri-v3-310m"
MODEL_DIR = Path(os.environ.get("VECMEMORI_EMBEDDING_MODEL", str(DEFAULT_MODEL)))

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.safetensors").exists(),
    reason="ruri-v3-310m not downloaded (run scripts/download_model.sh)",
)


def test_hybrid_search_and_contradict():
    tmpdir = tempfile.mkdtemp(prefix="vecmcp_embed_")
    os.environ["VECMEMORI_DB"] = os.path.join(tmpdir, "t.db")
    os.environ["VECMEMORI_FACT_LANGUAGE"] = "ja"
    os.environ["VECMEMORI_EMBEDDING_MODEL"] = str(MODEL_DIR)
    os.environ.pop("VECMEMORI_FTS_ONLY", None)  # full hybrid mode
    sys.path.insert(0, str(REPO_SRC))

    import vecmemori.mcp_server as ms  # noqa: E402

    try:
        assert ms.fact_store(
            action="add", content="ヒロイン: 桜井美咲。年齢19歳。黒髪ロング。大学生。",
            category="character", tags="美咲",
        )
        assert ms.fact_store(
            action="add", content="舞台は2045年の東京。第三次世界大戦後の復興都市。",
            category="worldbuilding",
        )

        # natural language, no shared keywords — embedding path
        r = json.loads(ms.fact_store(action="search", query="美咲ちゃんの髪型はどうだったっけ", limit=3))
        assert r["count"] == 1 and "黒髪ロング" in r["results"][0]["content"]

        # meaning-only paraphrase
        r2 = json.loads(ms.fact_store(action="search", query="戦争のあとで復興した街", limit=3))
        assert r2["count"] == 1 and "東京" in r2["results"][0]["content"]

        # contradict via paraphrase (20代後半社会人 vs 19歳大学生)
        r3 = json.loads(ms.fact_store(action="contradict", statement="美咲は20代後半の社会人だ", limit=3))
        assert r3["count"] == 1 and "19歳" in r3["results"][0]["content"]

        # probe
        r4 = json.loads(ms.fact_store(action="probe", entity="東京", limit=3))
        assert r4["count"] == 1

        store, _ = ms._get_state()
        n = store._conn.execute(
            "SELECT COUNT(*) FROM facts WHERE ruri_embedding IS NOT NULL").fetchone()[0]
        assert n == 2
    finally:
        store = getattr(ms, "_store", None)
        if store is not None:
            store.close()
            ms._store = None
            ms._retriever = None
        shutil.rmtree(tmpdir, ignore_errors=True)
