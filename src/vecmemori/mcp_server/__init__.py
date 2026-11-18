"""vecmemori MCP server — expose the fact memory to any MCP-capable agent.

Runs over stdio so every MCP host (Hermes, opencode, codex, gemini-cli,
OpenClaw, goose, ...) can share one local memory database.

Tools:
    fact_store     9 actions: add/search/probe/related/reason/contradict/update/remove/list
    fact_feedback  helpful/unhelpful rating that trains trust scores

Environment variables:
    VECMEMORI_DB               SQLite path (default: ~/.vecmemori/memory.db)
    VECMEMORI_EMBEDDING_MODEL  local SentenceTransformer model path
                               (default: ~/.vecmemori/models/ruri-v3-310m;
                               empty string disables embeddings)
    VECMEMORI_FTS_ONLY         "1"/"true" forces FTS-only mode (no model load)
    VECMEMORI_FTS_WEIGHT       FTS weight in hybrid search (default 0.40)
    VECMEMORI_EMBEDDING_WEIGHT embedding weight (default 0.60)
    VECMEMORI_MIN_TRUST        minimum trust for search results (default 0.3)
    VECMEMORI_DEFAULT_TRUST    trust for new facts (default 0.5)
    VECMEMORI_FACT_LANGUAGE    language guard for stored facts ("ja", "en",
                               ...; empty/unset = no guard)
    VECMEMORI_CATEGORY_SCOPE   optional operator-set category isolation
    VECMEMORI_READ_ONLY        "1"/"true" disables mutation tools

Usage:
    python -m vecmemori.mcp_server
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Literal

logger = logging.getLogger("vecmemori.mcp_server")


def _setup_logging() -> Path:
    db_path = Path(
        os.environ.get("VECMEMORI_DB", Path.home() / ".vecmemori" / "memory.db")
    ).expanduser()
    log_path = db_path.parent / "mcp_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    return log_path


_LOG_PATH = _setup_logging()

from vecmemori import MemoryStore, FactRetriever  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

# ---------------------------------------------------------------------------
# Language guard (subset of the Hermes adapter's guard, MCP-safe default)
# ---------------------------------------------------------------------------

_JA_PATTERN = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")  # kana + CJK ideographs
_KO_PATTERN = re.compile(r"[\uac00-\ud7af]")
_ZH_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def _normalize_language(value: object) -> str | None:
    if value is None:
        return None
    value = str(value).strip().lower()
    return value or None


def _language_guard_error(language: str | None, content: str) -> str | None:
    language = _normalize_language(language)
    if not language:
        return None
    if language == "ja":
        if not _JA_PATTERN.search(content):
            return "Fact content must contain Japanese text (fact_storage_language=ja)."
        if _KO_PATTERN.search(content):
            return "Fact content must not contain Korean text (fact_storage_language=ja)."
    elif language == "ko":
        if not _KO_PATTERN.search(content):
            return "Fact content must contain Korean text (fact_storage_language=ko)."
    elif language == "zh":
        if not _ZH_PATTERN.search(content):
            return "Fact content must contain Chinese text (fact_storage_language=zh)."
    elif language == "en":
        # Heuristic: reject facts that are mostly CJK characters.
        cjk = len(_JA_PATTERN.findall(content))
        if cjk > len(content) * 0.5:
            return "Fact content must not be mostly CJK text (fact_storage_language=en)."
    return None


def _validate_fact_write(language: str | None, content: str) -> str | None:
    return _language_guard_error(language, content)


# OR-expand natural-language searches while quoting each token. The MCP
# caller never supplies executable FTS syntax; drop Japanese particles first.
_JA_STOPWORDS = frozenset(
    "は が を に へ と で や か も の な だ です ます た て で に から まで "
    "これ それ あれ この その あの する した なる ある いる こと もの".split()
)
# Word token for FTS matching: letters/digits of any script, or CJK/kana chars
_WORD_TOKEN = re.compile(
    r"^[\w\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]+$", re.UNICODE
)


def _fts_query_expand(query: str) -> str:
    """Tokenize then OR-expand a natural-language query for FTS5 MATCH.

    Callers cannot inject FTS5 operators.
    Punctuation-only tokens ("?", "。", ...) and non-word tokens are dropped,
    and every kept token is double-quoted so reserved words (OR, NEAR) or
    digits cannot break the MATCH expression.
    """
    if not query:
        return query
    from vecmemori._tokenizer import tokenize_query

    tokens = re.findall(r"[\w\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]+",
                        tokenize_query(query), re.UNICODE)
    kept: list[str] = []
    for tok in tokens:
        if len(kept) >= 20:
            break
        if tok in _JA_STOPWORDS:
            continue
        if not _WORD_TOKEN.fullmatch(tok):
            continue
        if tok not in kept:
            kept.append(tok)
    if not kept:
        raise ValueError("query contains no searchable terms")
    if len(kept) == 1:
        return f'"{kept[0]}"'
    return " OR ".join(f'"{t}"' for t in kept)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


class _Config:
    db_path: str
    embedding_model: str | None
    fts_only: bool
    fts_weight: float
    embedding_weight: float
    min_trust: float
    default_trust: float
    fact_language: str | None
    category_scope: str | None
    read_only: bool


def _load_config() -> _Config:
    cfg = _Config()
    default_model = Path.home() / ".vecmemori" / "models" / "ruri-v3-310m"
    db = os.environ.get("VECMEMORI_DB", str(Path.home() / ".vecmemori" / "memory.db"))
    cfg.db_path = str(Path(db).expanduser())
    model_env = os.environ.get("VECMEMORI_EMBEDDING_MODEL", str(default_model))
    cfg.embedding_model = str(Path(model_env).expanduser()) if model_env else None
    cfg.fts_only = _env_bool("VECMEMORI_FTS_ONLY")
    cfg.fts_weight = _env_float("VECMEMORI_FTS_WEIGHT", 0.40)
    cfg.embedding_weight = _env_float("VECMEMORI_EMBEDDING_WEIGHT", 0.60)
    cfg.min_trust = _env_float("VECMEMORI_MIN_TRUST", 0.3)
    cfg.default_trust = _env_float("VECMEMORI_DEFAULT_TRUST", 0.5)
    cfg.fact_language = _normalize_language(os.environ.get("VECMEMORI_FACT_LANGUAGE", ""))
    cfg.category_scope = os.environ.get("VECMEMORI_CATEGORY_SCOPE", "").strip() or None
    cfg.read_only = _env_bool("VECMEMORI_READ_ONLY")
    return cfg


CONFIG = _load_config()

# ---------------------------------------------------------------------------
# Store / retriever (lazy singletons)
# ---------------------------------------------------------------------------

_store: MemoryStore | None = None
_retriever: FactRetriever | None = None


def _get_state() -> tuple[MemoryStore, FactRetriever]:
    global _store, _retriever
    if _store is None or _retriever is None:
        require_embeddings = not CONFIG.fts_only
        _store = MemoryStore(
            db_path=CONFIG.db_path,
            default_trust=CONFIG.default_trust,
            require_embeddings=require_embeddings,
        )
        _retriever = FactRetriever(
            store=_store,
            fts_weight=CONFIG.fts_weight,
            ruri_weight=0.0 if CONFIG.fts_only else CONFIG.embedding_weight,
            require_embeddings=require_embeddings,
        )
        logger.info(
            "vecmemori store ready: db=%s fts_only=%s log=%s",
            CONFIG.db_path, CONFIG.fts_only, _LOG_PATH,
        )
    return _store, _retriever


def _configure_embedder() -> None:
    if CONFIG.fts_only or not CONFIG.embedding_model:
        return
    try:
        from vecmemori._embedder import set_model_path, set_trust_remote_code

        set_model_path(CONFIG.embedding_model)
        set_trust_remote_code(_env_bool("VECMEMORI_EMBEDDING_TRUST_REMOTE_CODE"))
        logger.info("embedding model configured: %s", CONFIG.embedding_model)
    except Exception as exc:  # non-fatal: FTS path still works
        logger.warning("failed to configure embedding model %r: %s", CONFIG.embedding_model, exc)


_configure_embedder()

# ---------------------------------------------------------------------------
# fact_store handler (ported from the Hermes adapter, actions preserved)
# ---------------------------------------------------------------------------

MAX_TOOL_RESULTS = 50
_MAX_QUERY_CHARS = 256


def _effective_category(requested: str | None) -> str | None:
    """Operator scope wins over requests; clients cannot escape their category."""
    scope = CONFIG.category_scope
    if scope is not None and requested is not None and requested != scope:
        raise ValueError("category outside configured scope")
    return scope if scope is not None else requested


def _checked_limit(args: dict) -> int:
    limit = args.get("limit", 10)
    if type(limit) is not int or not (1 <= limit <= MAX_TOOL_RESULTS):
        raise ValueError(f"limit must be an integer from 1 to {MAX_TOOL_RESULTS}")
    return limit


def _checked_query(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_QUERY_CHARS:
        raise ValueError(f"query must be nonempty text of at most {_MAX_QUERY_CHARS} characters")
    return value


def _handle_fact_store(args: dict) -> str:
    try:
        action = args["action"]
        if CONFIG.read_only and action in ("add", "update", "remove"):
            return _tool_error("read-only memory server: mutation disabled")
        category = _effective_category(args.get("category"))
        if action in ("search", "probe", "related", "reason", "contradict", "list"):
            limit = _checked_limit(args)
        store, retriever = _get_state()

        if action == "add":
            content = args["content"]
            validation_error = _validate_fact_write(CONFIG.fact_language, content)
            if validation_error:
                return _tool_error(validation_error)
            fact_id = store.add_fact(
                content,
                category=category or "general",
                tags=args.get("tags") or "",
            )
            return _json({"fact_id": fact_id, "status": "added"})

        if action == "search":
            results = _hybrid_search(
                retriever,
                _checked_query(args["query"]),
                category=category,
                min_trust=float(args.get("min_trust", CONFIG.min_trust)),
                limit=limit,
            )
            return _json({"results": results, "count": len(results)})

        if action == "probe":
            entity = _checked_query(args.get("entity", ""))
            results = _hybrid_search(
                retriever,
                entity,
                category=category,
                min_trust=float(args.get("min_trust", CONFIG.min_trust)),
                limit=limit,
            )
            return _json({"results": results, "count": len(results)})

        if action == "related":
            entity = _checked_query(args.get("entity", ""))
            results = _hybrid_search(
                retriever,
                entity,
                category=category,
                min_trust=float(args.get("min_trust", CONFIG.min_trust)),
                limit=limit,
            )
            return _json({"results": results, "count": len(results)})

        if action == "reason":
            entities = args.get("entities", [])
            if not isinstance(entities, list) or not (1 <= len(entities) <= 8):
                return _tool_error("reason requires 1–8 entities")
            entities = [_checked_query(entity) for entity in entities]
            # Not symbolic reasoning: pragmatic multi-query semantic recall,
            # unioned by fact_id (same policy as the Hermes adapter).
            seen: set[int] = set()
            merged: list[dict] = []
            for entity in entities:
                for result in _hybrid_search(
                    retriever,
                    entity,
                    category=category,
                    min_trust=float(args.get("min_trust", CONFIG.min_trust)),
                    limit=limit,
                ):
                    fid = result.get("fact_id")
                    if fid not in seen:
                        seen.add(fid)
                        merged.append(result)
            merged = merged[:limit]
            return _json({"results": merged, "count": len(merged)})

        if action == "contradict":
            # Improvement over the Hermes adapter's stub (which only listed
            # recent facts): retrieve facts semantically close to the
            # statement so the caller can inspect likely conflicts directly.
            statement = _checked_query(args.get("statement", ""))
            results = _hybrid_search(
                retriever,
                statement,
                category=category,
                min_trust=0.0,
                limit=limit,
            )
            return _json({
                "results": results,
                "count": len(results),
                "note": ("Candidates similar to the statement. Compare manually — "
                         "automated contradiction detection is heuristic."),
            })

        if action == "update":
            trust_delta = (float(args["trust_delta"])
                           if args.get("trust_delta") is not None else None)
            if trust_delta is not None and (
                not math.isfinite(trust_delta) or abs(trust_delta) > 0.1
            ):
                return _tool_error("trust_delta must be between -0.1 and 0.1")
            if "content" in args and args.get("content") is not None:
                validation_error = _validate_fact_write(CONFIG.fact_language, args["content"])
                if validation_error:
                    return _tool_error(validation_error)
            updated = store.update_fact(
                int(args["fact_id"]),
                content=args.get("content"),
                trust_delta=trust_delta,
                tags=args.get("tags"),
                category=args.get("category"),
                scope=CONFIG.category_scope,
            )
            return _json({"updated": updated})

        if action == "remove":
            removed = store.remove_fact(
                int(args["fact_id"]), scope=CONFIG.category_scope
            )
            return _json({"removed": removed})

        if action == "list":
            facts = store.list_facts(
                category=category,
                min_trust=float(args.get("min_trust", 0.0)),
                limit=limit,
            )
            return _json({"facts": facts, "count": len(facts)})

        return _tool_error(f"Unknown action: {action}")

    except KeyError as exc:
        return _tool_error(f"Missing required argument: {exc}")
    except Exception as exc:
        logger.exception("fact_store failed")
        return _tool_error(str(exc))


def _handle_fact_feedback(args: dict) -> str:
    try:
        if CONFIG.read_only:
            return _tool_error("read-only memory server: feedback disabled")
        fact_id = int(args["fact_id"])
        if args["action"] not in ("helpful", "unhelpful"):
            return _tool_error("unknown feedback action")
        helpful = args["action"] == "helpful"
        store, _ = _get_state()
        result = store.record_feedback(
            fact_id, helpful=helpful, scope=CONFIG.category_scope
        )
        return _json(result)
    except KeyError as exc:
        return _tool_error(f"Missing required argument: {exc}")
    except Exception as exc:
        logger.exception("fact_feedback failed")
        return _tool_error(str(exc))


def _json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _hybrid_search(retriever: FactRetriever, query: str, *, category: str | None,
                   min_trust: float, limit: int) -> list[dict]:
    """Search with an OR-expanded FTS query while the raw text drives embeddings."""
    return retriever.search(
        query,
        category=category,
        min_trust=min_trust,
        limit=limit,
        fts_query=_fts_query_expand(query),
    )


def _tool_error(message: str) -> str:
    return _json({"error": message})


# ---------------------------------------------------------------------------
# Typed tool wrappers — the MCP SDK v2 derives the JSON schema for clients
# from these annotations, so keep them descriptive.
# ---------------------------------------------------------------------------

Action = Literal[
    "add", "search", "probe", "related", "reason",
    "contradict", "update", "remove", "list",
]


def fact_store(  # noqa: ANN201 — returns a JSON string
    action: Action,
    content: str | None = None,
    query: str | None = None,
    entity: str | None = None,
    entities: list[str] | None = None,
    statement: str | None = None,
    fact_id: int | None = None,
    category: str | None = None,
    tags: str | None = None,
    trust_delta: float | None = None,
    min_trust: float | None = None,
    limit: int | None = None,
) -> str:
    """Long-term fact memory.

    add: save a durable fact (content required; category/tags optional).
    search: semantic + FTS recall (query required).
    probe / related: all facts around one entity.
    reason: union of facts across several entities.
    contradict: facts similar to a statement — check before asserting a
    fact as established truth.
    update / remove / list: maintenance (fact_id required for update/remove).
    """
    args = {
        k: v for k, v in {
            "action": action, "content": content, "query": query,
            "entity": entity, "entities": entities, "statement": statement,
            "fact_id": fact_id, "category": category, "tags": tags,
            "trust_delta": trust_delta, "min_trust": min_trust, "limit": limit,
        }.items() if v is not None
    }
    return _handle_fact_store(args)


def fact_feedback(
    action: Literal["helpful", "unhelpful"],
    fact_id: int,
) -> str:
    """Rate a fact after using it: helpful/unhelpful. Good facts rise, bad facts sink."""
    return _handle_fact_feedback({"action": action, "fact_id": fact_id})


# ---------------------------------------------------------------------------
# Server wiring
# ---------------------------------------------------------------------------

def create_server() -> FastMCP:
    server = FastMCP(
        name="vecmemori-plus",
        instructions=(
            "Local fact memory (SQLite + FTS5 + optional Japanese embeddings). "
            "Store durable facts with fact_store action=add, recall with "
            "search/probe/reason, and run contradict before asserting an "
            "established fact. Rate used facts with fact_feedback."
        ),
    )
    server.add_tool(
        fact_store,
        name="fact_store",
        description=(
            "Long-term fact memory. add: save a durable fact (character "
            "settings, decisions, foreshadowing). search/probe/reason: recall. "
            "contradict: find facts similar to a statement before treating it "
            "as truth. update/remove/list: maintenance."
        ),
    )
    server.add_tool(
        fact_feedback,
        name="fact_feedback",
        description=(
            "Rate a fact after using it: helpful/unhelpful. Good facts rise, "
            "bad facts sink (trust scoring)."
        ),
    )
    return server


def main() -> int:
    server = create_server()
    logger.info("starting vecmemori MCP server (stdio)")
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
