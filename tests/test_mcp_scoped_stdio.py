"""Exercise operator-side category scopes over real MCP stdio processes."""
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_SRC = Path(__file__).resolve().parents[1] / "src"


async def session_calls(db, scope, *requests, read_only=False):
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_SRC) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "VECMEMORI_DB": str(db),
        "VECMEMORI_FTS_ONLY": "1",
        "VECMEMORI_EMBEDDING_MODEL": "",
        "VECMEMORI_CATEGORY_SCOPE": scope,
        "VECMEMORI_READ_ONLY": "1" if read_only else "0",
    }
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "vecmemori.mcp_server"],
        env=env,
    )
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await asyncio.wait_for(session.initialize(), timeout=15)
            results = []
            for tool, args in requests:
                res = await asyncio.wait_for(session.call_tool(tool, args), timeout=15)
                results.append(json.loads(res.content[0].text))
            return results


async def exercise(db):
    (a,) = await session_calls(
        db, "agent_a",
        ("fact_store", {"action": "add", "content": "SHARED_TEST_PROTECTED_A"}),
    )
    (b,) = await session_calls(
        db, "agent_b",
        ("fact_store", {"action": "add", "content": "SHARED_TEST_PROTECTED_B"}),
    )
    a_id, b_id = a["fact_id"], b["fact_id"]
    assert a_id != b_id

    a_list, a_foreign, a_feedback, a_delete = await session_calls(
        db, "agent_a",
        ("fact_store", {"action": "list"}),
        ("fact_store", {"action": "update", "fact_id": b_id,
                         "content": "UNAUTHORIZED_MODIFICATION"}),
        ("fact_feedback", {"action": "helpful", "fact_id": b_id}),
        ("fact_store", {"action": "remove", "fact_id": b_id}),
    )
    assert a_list["count"] == 1
    assert a_list["facts"][0]["fact_id"] == a_id
    assert a_foreign == {"updated": False}
    assert "error" in a_feedback
    assert a_delete == {"removed": False}

    b_list, b_readonly_add, b_readonly_feedback = await session_calls(
        db, "agent_b",
        ("fact_store", {"action": "list"}),
        ("fact_store", {"action": "add", "content": "UNAUTHORIZED_ADDITION"}),
        ("fact_feedback", {"action": "helpful", "fact_id": b_id}),
        read_only=True,
    )
    assert b_list["count"] == 1
    assert b_list["facts"][0]["fact_id"] == b_id
    assert "error" in b_readonly_add
    assert "error" in b_readonly_feedback
    return {"agent_a": a_id, "agent_b": b_id}


def test_real_stdio_separate_scopes(tmp_path):
    assert asyncio.run(exercise(tmp_path / "shared.sqlite")) == {
        "agent_a": 1, "agent_b": 2,
    }
