from __future__ import annotations

import asyncio
import os
import sys
import time
from collections import deque

import openai
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()

TURNS = [
    "My name is Sara.",
    "I'm working on a Python data pipeline.",
    "Constraint: use CUDA 12.4 for any code you suggest.",
    "It reads CSV files and writes Parquet.",
    "It runs on a 4-core VM and takes 20 minutes.",
    "I want it under 5 minutes.",
    "The bottleneck is CSV parsing.",
    "I tried pandas read_csv with chunksize.",
    "That helped a little, not much.",
    "What should I try next?",
]

OTPM = 1000          # free-tier output tokens/minute for this model
WINDOW_S = 60.0
SAFETY = 0.85        # leave headroom for the provider's own accounting
MAX_OUT = 512        # if a turn lands exactly here, it was truncated


def last_usage(messages):
    for m in reversed(messages):
        u = getattr(m, "usage_metadata", None)
        if u:
            return u
    return {}


class Pacer:
    """Keeps actual output tokens under the per-minute budget."""

    def __init__(self, otpm: int, window: float = WINDOW_S, safety: float = SAFETY):
        self.budget = int(otpm * safety)
        self.window = window
        self.events: deque[tuple[float, int]] = deque()

    def _used(self, now: float) -> int:
        while self.events and now - self.events[0][0] > self.window:
            self.events.popleft()
        return sum(t for _, t in self.events)

    async def acquire(self) -> None:
        while True:
            now = time.monotonic()
            used = self._used(now)
            if used < self.budget or not self.events:
                return
            wait = self.window - (now - self.events[0][0]) + 0.5
            print(f"  [pace] {used}/{self.budget} output tokens used; sleeping {wait:.1f}s")
            await asyncio.sleep(wait)

    def record(self, tokens: int) -> None:
        self.events.append((time.monotonic(), tokens))


async def invoke_with_retry(agent, payload, config, attempts: int = 6):
    delay = 5.0
    for i in range(attempts):
        try:
            return await agent.ainvoke(payload, config=config)
        except openai.RateLimitError as e:
            if i == attempts - 1:
                raise
            # the 429 body usually states the wait; back off generously anyway
            print(f"  [429] {type(e).__name__}; retrying in {delay:.0f}s")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)


async def main() -> None:
    model = ChatOpenAI(
        model=os.environ["MODEL"],
        timeout=60,          # 20 s is tight for a reasoning model at 512 tokens
        max_retries=0,       # retries handled in invoke_with_retry
        max_tokens=MAX_OUT,
    )

    client = MultiServerMCPClient({
        "basic": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "basic_agent.mcp_server"],
        }
    })
    tools = await client.get_tools()
    agent = create_agent(model, tools=tools)

    pacer = Pacer(OTPM)
    history = []

    for turn, text in enumerate(TURNS, 1):
        history.append(("user", text))
        await pacer.acquire()
        result = await invoke_with_retry(
            agent,
            {"messages": history},
            {"recursion_limit": 12},
        )
        history = result["messages"]

        u = last_usage(result["messages"])
        out = u.get("output_tokens", 0)
        pacer.record(out)

        flag = "  TRUNCATED" if out >= MAX_OUT else ""
        print(
            f"turn {turn:2}  "
            f"in={u.get('input_tokens', 0):>7,}  "
            f"out={out:>5,}  "
            f"msgs={len(result['messages'])}{flag}"
        )


if __name__ == "__main__":
    asyncio.run(main())
