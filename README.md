<h1 align="center">vecmemori-plus</h1>
<p align="center">
  <em>Local fact memory for AI agents — SQLite + FTS5 + neural embeddings + MCP server for multi-agent shared memory</em>
  <br>
  <a href="README.ja.md">日本語 README</a>
</p>

---

**vecmemori-plus** is a local, persistent fact memory engine for AI agents.
(The import module name stays `vecmemori` for compatibility; the pip package is `vecmemori-plus`.) It stores short facts such as user preferences, project decisions, and environment details in SQLite, then retrieves relevant facts with a dual strategy:

- **FTS5 full-text search** for keyword precision
- **Neural embeddings** for semantic similarity

Embeddings are a core requirement. vecmemori-plus is not intended to be only a keyword-search database.

> **Status:** Alpha. The core API is usable, but packaging and Hermes Agent integration may still change before a stable release.

## About this fork

This repository is a fork of [`iwaan10000vr/vecmemori`](https://github.com/iwaan10000vr/vecmemori) — thank you, [@iwaan](https://github.com/iwaan10000vr), for building the original Japanese-first local memory engine. All credit for the core design (SQLite + FTS5 + neural embeddings, fugashi tokenization, ruri-v3 as the default model) belongs to the upstream project.

**Why we forked.** We run long-form novel projects across several AI agents (Hermes Agent, opencode, goose, OpenClaw) and wanted **one shared memory database** — the same characters, worldbuilding, and foreshadowing visible from every agent. The upstream project needed an update to its Hermes Agent-only integration, and we also had feature extensions in mind — such as supporting other agents — so we maintain an extended distribution in this repository.

**What this fork adds (v0.2.0):**

- **MCP stdio server** (`vecmemori.mcp_server`) — exposes `fact_store` (9 actions) and `fact_feedback` to any MCP-capable agent. One config line per agent, one shared SQLite DB. See [MCP server](#mcp-server-multi-agent-shared-memory).
- **FTS5 AND-trap fix for Japanese** — natural-language queries are tokenized, Japanese particles are dropped, and the query is OR-expanded (with quoting) so full-sentence queries no longer zero out recall.
- **`fts_query` parameter** on `FactRetriever.search()` — the FTS MATCH expression is decoupled from the embedding query, so query expansion never pollutes the semantic path.
- **`contradict` improvement** — now searches facts similar to a statement (the upstream version only listed recent facts as a stub), so agents can check an assertion before writing it as truth.
- **FTS-only mode** (`VECMEMORI_FTS_ONLY=1`) — run without the 1.2 GB embedding model for lightweight agents.
- **Environment-variable configuration** — `VECMEMORI_DB`, `VECMEMORI_FACT_LANGUAGE`, etc. The MCP path has no Hermes dependency.
- **Setup snippets** for opencode / goose / OpenClaw (connection-tested), and an end-to-end stdio test (`tests/test_mcp_e2e.py`).

The Hermes Agent provider adapter is kept and unchanged in behavior.

## Privacy model

The core library stores data locally in SQLite and uses a local SentenceTransformer-compatible embedding model by default. Data does not leave your device when you use the core library with a local model.

Some integrations can intentionally send text to configured LLM providers:

- Hermes Agent `auto_extract` may send recent conversation messages to an LLM to extract durable facts.
- Hermes Agent `retrieval_planner` / planner injection may send conversation context to an LLM to generate better retrieval queries.
- A remote embedding provider, if you explicitly configure one outside the default local model flow, may receive text for embedding.

Do not enable those options unless the configured provider is acceptable for your privacy requirements.

## Installation

```bash
pip install vecmemori-plus
pip install vecmemori-plus[ja]       # Japanese FTS5 tokenization with fugashi + unidic-lite
pip install vecmemori-plus[hermes]   # adapter dependencies for Hermes Agent environments
pip install vecmemori-plus[all]      # Japanese + Hermes adapter dependencies
```

Install a local embedding model before writing/searching facts:

```bash
bash scripts/download_model.sh
```

Default model path:

```text
~/.cache/vecmemori/models/ruri-v3-310m
```

The default model is [`cl-nagoya/ruri-v3-310m`](https://huggingface.co/cl-nagoya/ruri-v3-310m), which is Apache-2.0 licensed and works well for Japanese.

## Quick Start

```python
from vecmemori import MemoryStore, FactRetriever

store = MemoryStore(db_path="memory.db")
store.add_fact("User prefers dark mode in all applications", category="user_pref")
store.add_fact("Project uses pytest for testing", category="project")

retriever = FactRetriever(store=store)
results = retriever.search("testing preferences", limit=5)

for r in results:
    print(f"[{r['trust_score']:.2f}] {r['content']}")
```

For explicit tests or diagnostics without embeddings, pass `require_embeddings=False` to both `MemoryStore` and `FactRetriever`. This is not the recommended production mode.

## Japanese FTS5 support

SQLite's default `unicode61` tokenizer does not segment Japanese text well. Install the Japanese extra to pre-tokenize text with [fugashi](https://github.com/polm/fugashi) / MeCab:

```bash
pip install vecmemori-plus[ja]
python -c "from vecmemori._tokenizer import has_tokenizer; print('Japanese FTS5:', has_tokenizer())"
```

Neural embedding search is still the main semantic recall path. Japanese FTS5 is an additional keyword-precision signal.

## Features

- **Dual-strategy search** — FTS5 + neural embeddings, weighted by default at 0.40 / 0.60
- **Required semantic embeddings** — local SentenceTransformer-compatible model
- **Japanese support** — ruri-v3 embeddings plus optional fugashi tokenization for FTS5
- **Entity extraction** — lightweight entity extraction from fact content
- **Trust scoring** — helpful/unhelpful feedback adjusts fact reliability
- **Temporal decay** — optional recency weighting
- **Model swapping** — configurable embedding dimension and query/document prefixes
- **Hermes Agent integration** — memory provider adapter, `fact_store` tool, prefetch, optional LLM planner/extraction
- **MCP server** — share one memory database across MCP-capable agents (opencode, goose, OpenClaw, ...)

## Python API

```python
from vecmemori import MemoryStore, FactRetriever

store = MemoryStore(db_path="memory.db")
fact_id = store.add_fact("GPU: RTX 5060 Ti 16GB", category="tool", tags="hardware,gpu")
store.record_feedback(fact_id=fact_id, helpful=True)

retriever = FactRetriever(store=store)
results = retriever.search("gpu memory", limit=5)

store.rebuild_all_embeddings()  # after changing embedding model/config
store.close()
```

`probe`, `reason`, and `contradict` are Hermes `fact_store` tool actions, not methods on the standalone `FactRetriever` Python API.

## Hermes Agent integration

`vecmemori-plus[hermes]` installs the adapter dependencies (`PyYAML`, `httpx`) needed by the Hermes plugin module. It does **not** install Hermes Agent itself. Use this extra inside an environment where Hermes Agent is already installed/configured.

### Install into Hermes Agent from GitHub

Until vecmemori has a dedicated Hermes plugin installer, use the source checkout as the plugin directory and install the Python package into the same environment where Hermes runs:

```bash
git clone https://github.com/iwaan10000vr/vecmemori.git
cd vecmemori
python -m pip install -e ".[hermes,ja]"
bash scripts/download_model.sh

mkdir -p ~/.hermes/plugins
ln -s "$(pwd)/src/vecmemori/hermes" ~/.hermes/plugins/vecmemori
hermes config set memory.provider vecmemori
hermes config set plugins.vecmemori.embedding_model "$HOME/.cache/vecmemori/models/ruri-v3-310m"
```

Optional local-first settings:

```bash
hermes config set plugins.vecmemori.auto_extract true
hermes config set plugins.vecmemori.fact_storage_language ja   # language used to write stored fact content
hermes config set plugins.vecmemori.retrieval_planner false
hermes config set plugins.vecmemori.embedding_trust_remote_code false
```

Restart Hermes Agent after changing plugin code or memory provider settings.

Lifecycle when used as a Hermes memory provider:

1. **Every user message:** prefetch searches facts and injects top matches into context.
2. **Tool call:** `fact_store` can add/search/update/remove facts explicitly.
3. **Built-in memory mirroring:** Hermes `memory` writes can be mirrored into vecmemori.
4. **Session end:** optional `auto_extract` can ask a configured LLM to extract durable facts.
5. **Planner:** optional `retrieval_planner` can ask a configured LLM to generate multiple retrieval queries.

Hermes-only tool actions:

- `add`, `search`, `probe`, `related`, `reason`, `contradict`, `update`, `remove`, `list`

The `reason`/`contradict` actions are pragmatic semantic recall helpers, not symbolic algebraic proof or guaranteed contradiction detection.

## MCP server (multi-agent shared memory)

The `vecmemori.mcp_server` module exposes the same fact memory to **any MCP host** — Hermes, opencode, goose, OpenClaw, and other MCP-compatible agents. All agents share one SQLite database, one embedding model, one truth.

> **Connection-tested:** opencode 1.18, goose 1.49, OpenClaw 2026.9 (write + cross-agent read + feedback verified against one shared DB).

### Install and run

```bash
pip install -e ".[ja]"          # fugashi tokenization recommended for Japanese
bash scripts/download_model.sh  # ruri-v3-310m for semantic search (optional)

# quick test (FTS-only, no model):
VECMEMORI_FTS_ONLY=1 python -m vecmemori.mcp_server
```

Configuration via environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `VECMEMORI_DB` | `~/.vecmemori/memory.db` | SQLite database path (point ALL agents here) |
| `VECMEMORI_EMBEDDING_MODEL` | `~/.vecmemori/models/ruri-v3-310m` | local SentenceTransformer model path |
| `VECMEMORI_FTS_ONLY` | off | `1` disables embeddings (no model load, FTS5 keyword search only) |
| `VECMEMORI_FACT_LANGUAGE` | unset | language guard for stored facts (`ja`, `en`, `ko`, `zh`) |
| `VECMEMORI_MIN_TRUST` | `0.3` | minimum trust filter for search results |

Tools exposed: `fact_store` (actions: add / search / probe / related / reason / contradict / update / remove / list) and `fact_feedback` (helpful / unhelpful).

### Agent configuration snippets

**Hermes Agent** — prefer the native provider (auto prefetch + sync; see below). To use the MCP server instead, add it as an MCP server pointing at the venv python.

**opencode** (`~/.config/opencode/opencode.json`):

```jsonc
{
  "mcp": {
    "vecmemori": {
      "type": "local",
      "command": ["C:/Users/<you>/path/to/python.exe", "-m", "vecmemori.mcp_server"],
      "enabled": true,
      "environment": {
        "VECMEMORI_DB": "C:/Users/<you>/.vecmemori/memory.db",
        "VECMEMORI_FACT_LANGUAGE": "ja"
      }
    }
  }
}
```

**goose / OpenClaw** — register the same stdio command (`python -m vecmemori.mcp_server`) as an MCP server; goose via `goose configure` extensions, OpenClaw via Control UI → Settings → MCP (or `mcp.servers` in config).

> **Note:** the MCP server is additive — the Hermes Agent native provider and the MCP server can share the same `VECMEMORI_DB`. Use the native provider inside Hermes (turn prefetch + LLM planner), and MCP everywhere else.

### contradict action

`contradict` retrieves facts semantically close to a statement so the caller can verify an assertion before writing new facts. It is a recall-based heuristic, not logic reasoning: inspect the returned candidates and judge the conflict yourself.

## Configuration

Common options:

- `db_path`: SQLite database path, default `memory.db` in standalone use
- `default_trust`: initial trust score, default `0.5`
- `fts_weight`: keyword score weight, default `0.40`
- `ruri_weight`: semantic embedding score weight, default `0.60` (legacy config key name)
- `prefetch_limit`: Hermes facts injected per turn, default `5`
- `auto_extract`: Hermes LLM extraction on session end, default depends on plugin config
- `fact_storage_language`: language used to write stored fact content (for example `ja` or `en`). If unset/null/auto, vecmemori refuses fact writes and the agent should ask the user which language to use first. Proper nouns, code, paths, product names, and natural technical terms are preserved.
- `retrieval_planner`: Hermes LLM multi-query retrieval, default `false`
- `embedding_model`: local embedding model path
- `embedding_trust_remote_code`: allow Hugging Face custom model code, default `false`

## Model Swap

```python
from vecmemori._embedder import set_config, set_model_path

set_config(dimension=384, query_prefix="", doc_prefix="")
set_model_path("/path/to/all-MiniLM-L6-v2")

from vecmemori import MemoryStore
store = MemoryStore("memory.db")
store.rebuild_all_embeddings()
```

Recommended examples:

- `cl-nagoya/ruri-v3-310m` — Japanese, 768 dim, ~1.2 GB
- `sentence-transformers/all-MiniLM-L6-v2` — English, 384 dim, small
- `BAAI/bge-large-en-v1.5` — English, 1024 dim
- `intfloat/multilingual-e5-large` — multilingual, 1024 dim

## Architecture

```text
User Query
    │
    ▼
┌──────────────────────────┐
│ 1. FTS5 / BM25           │  weight=0.40 — keyword match
│ 2. Neural cosine sim     │  weight=0.60 — semantic similarity
└──────────┬───────────────┘
           ▼
    Relevance × Trust × Decay
           ▼
    Sorted results
```

## Development

```bash
pip install -e ".[dev,ja]"
python -m pytest -q
python -m build --sdist --wheel
python -m twine check dist/*
check-manifest
```

## License

MIT — see [LICENSE](LICENSE).

Third-party license information is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Attribution for the Hermes Agent / Holographic memory plugin origin is included in [NOTICE](NOTICE) and the third-party notices.

## Acknowledgments

**vecmemori-plus** exists thanks to [@iwaan](https://github.com/iwaan10000vr) — the upstream [`vecmemori`](https://github.com/iwaan10000vr/vecmemori) author. This fork is built directly on that work: the core store/retriever design, the Japanese FTS5 approach with fugashi, and the choice of cl-nagoya's ruri-v3 as the default embedding model all come from upstream. Thank you for the clean, well-documented foundation.

Going one layer deeper, upstream vecmemori itself began as a fork/rewrite of the MIT-licensed `holographic` memory plugin bundled with [Hermes Agent](https://github.com/NousResearch/hermes-agent) by Nous Research — our fork is the third generation of that lineage.

Thanks also to the maintainers of Hermes Agent, SQLite, NumPy, sentence-transformers, PyTorch, fugashi/MeCab, UniDic, and cl-nagoya's ruri models.