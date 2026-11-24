from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()


async def main() -> None:
    text = " ".join(sys.argv[1:]).strip() or "My name is Sara. Remember it."
    model = ChatOpenAI(
        model=os.environ["MODEL"],
        timeout=20, max_retries=0, max_tokens=512,
    )
    client = MultiServerMCPClient({
        "basic": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "basic_agent.mcp_server"],
        },
        "memory": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "basic_agent.memory_server"],
        },
    })
    tools = await client.get_tools()
    agent = create_agent(model, tools=tools)
    r = await agent.ainvoke(
        {"messages": [("user", text)]},
        config={"recursion_limit": 12},
    )
    print(r["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
