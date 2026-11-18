"""E2E test: launch the vecmemori MCP server over stdio and exercise it as a client.

Runs the server with sys.executable in FTS-only mode (no embedding model needed),
so it works from a source checkout or an installed package.

Usage:
    python tests/test_mcp_e2e.py
"""
import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parent.parent / "src"


async def run() -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    tmpdir = tempfile.mkdtemp(prefix="vecmcp_e2e_")
    db = os.path.join(tmpdir, "e2e.db")
    env = {
        **os.environ,
        "VECMEMORI_DB": db,
        "VECMEMORI_FTS_ONLY": "1",
        "VECMEMORI_FACT_LANGUAGE": "ja",
        "VECMEMORI_EMBEDDING_MODEL": "",
        "PYTHONPATH": str(REPO_SRC) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "vecmemori.mcp_server"], env=env
    )
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=30)
                tools = await session.list_tools()
                names = [t.name for t in tools.tools]
                assert names == ["fact_store", "fact_feedback"], f"unexpected tools: {names}"

                r = await asyncio.wait_for(session.call_tool(
                    "fact_store",
                    {"action": "add", "content": "ヒロイン: 桜井美咲。年齢19歳。黒髪ロング",
                     "category": "character"},
                ), 30)
                assert '"status": "added"' in r.content[0].text, r.content[0].text

                # FTS-only validates exact token recall, not semantic paraphrases.
                r2 = await asyncio.wait_for(session.call_tool(
                    "fact_store", {"action": "search", "query": "桜井美咲", "limit": 5}
                ), 30)
                assert "桜井美咲" in r2.content[0].text, r2.content[0].text

                r3 = await asyncio.wait_for(session.call_tool(
                    "fact_feedback", {"action": "helpful", "fact_id": 1}
                ), 30)
                assert "new_trust" in r3.content[0].text, r3.content[0].text

                print("TOOLS:", names)
                print("ADD/SEARCH/FEEDBACK: OK")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mcp_stdio() -> None:
    """Ensure the published MCP executable runs during pytest, too."""
    asyncio.run(run())


if __name__ == "__main__":
    asyncio.run(run())
    print("E2E OK")
