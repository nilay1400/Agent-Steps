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
    model = ChatOpenAI(
        model=os.environ["MODEL"],
        timeout=20,
        max_retries=0,
        max_tokens=512,
    )

    client = MultiServerMCPClient(
        {
            "basic": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "basic_agent.mcp_server"],
            }
        }
    )

    tools = await client.get_tools()
    print("Loaded MCP tools:", [t.name for t in tools])

    agent = create_agent(model, tools=tools)

    question = (
        "Use the add tool for 47832 + 91347, "
        "tell me the weather in Paris, "
        "and use run_python to compute 2**20."
    )

    result = await agent.ainvoke(
        {"messages": [("user", question)]},
        config={"recursion_limit": 12},
    )

    for message in result["messages"]:
        print(message)


if __name__ == "__main__":
    asyncio.run(main())
