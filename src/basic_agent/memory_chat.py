from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


async def main() -> None:
    model = ChatOpenAI(
        model=os.environ["MODEL"],
        timeout=20, max_retries=0, max_tokens=512,
    )
    client = MultiServerMCPClient({
        "basic": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "basic_agent.mcp_server"],
        }
    })
    tools = await client.get_tools()

    agent = create_agent(model, tools=tools, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "student-42"}}

    await agent.ainvoke({"messages": [("user", "My name is Sara.")]}, config=cfg)
    r = await agent.ainvoke({"messages": [("user", "What is my name?")]}, config=cfg)
    print(r["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
