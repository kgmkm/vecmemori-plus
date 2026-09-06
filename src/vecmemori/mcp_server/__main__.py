"""python -m vecmemori.mcp_server — stdio entry point for MCP hosts."""
import sys

from vecmemori.mcp_server import main

if __name__ == "__main__":
    sys.exit(main())
