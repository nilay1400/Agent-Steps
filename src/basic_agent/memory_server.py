from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("memory")

MEMORY = Path("memory.json")


@mcp.tool()
def remember(
    fact: Annotated[str, Field(description="A durable fact about the user or project")]
) -> str:
    """Save a lasting fact. Use when the user tells you something durable."""
    facts = json.loads(MEMORY.read_text()) if MEMORY.exists() else []
    facts.append(fact)
    MEMORY.write_text(json.dumps(facts, indent=2))
    return f"saved ({len(facts)} facts)"


@mcp.tool()
def recall() -> str:
    """Return every saved fact. Call BEFORE answering any question
    about the user or the project."""
    return MEMORY.read_text() if MEMORY.exists() else "[]"


if __name__ == "__main__":
    mcp.run()
