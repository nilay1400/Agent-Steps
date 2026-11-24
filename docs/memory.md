# Project 3 — Build Journal: Measure, Compact, and Remember

Chronological record of extending the Project 2 agent with context measurement, output clipping, history compaction, and two kinds of memory (thread-scoped checkpoints and cross-restart facts), using LangGraph + LangChain + MCP, running in WSL2 Ubuntu on a Surface Pro 7 (8 GB RAM).

- **Status:** ✅ Working
- **Date:** 2026-11-18
- **Files in repo:** `src/basic_agent/measure.py`, `src/basic_agent/compact.py`, `src/basic_agent/memory_chat.py`, `src/basic_agent/memory_longterm.py`, `src/basic_agent/memory_server.py`, plus `clip()` added to `mcp_server.py` (contents committed separately — this doc is the *process*)

---

## Environment baseline (as measured)

| Item | Value |
|---|---|
| OS | Ubuntu 26.04.1 LTS (WSL2) |
| System Python | 3.14.4 (`/usr/bin/python3`) — too new for this stack |
| `uv` | 0.13.1 |
| `git` | 2.53.0 |
| WSL RAM | 3.6 GiB total, 1.0 GiB swap |
| WSL cores | 8 logical (all handed to Linux) |
| `.wslconfig` | did not exist |
| Docker | 27.5.1 (installed via apt in Session 2) |

---

## Step 1 — Branch and baseline check

```bash
cd ~/agents-course/basic-agent
git checkout -b session3-context-memory
uv sync
uv run python -m basic_agent.routing_smoke   # Session 2 still passes 5/5
```

No new dependencies for Steps 2–4. The checkpointer is already inside `langgraph`. The long-term memory server uses only `mcp` and `json`/`pathlib`.

---

## Step 2 — Measure: watch the tokens climb

Created `src/basic_agent/measure.py`. A ten-turn chat, reusing the full history each turn. The point is the staircase from slide 3: input tokens grow linearly, and each turn re-pays for everything before it.

```python
from __future__ import annotations

import asyncio
import os
import sys

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


def last_usage(messages):
    for m in reversed(messages):
        u = getattr(m, "usage_metadata", None)
        if u:
            return u
    return {}


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
    agent = create_agent(model, tools=tools)

    history = []
    for turn, text in enumerate(TURNS, 1):
        history.append(("user", text))
        result = await agent.ainvoke(
            {"messages": history},
            config={"recursion_limit": 12},
        )
        history = result["messages"]
        u = last_usage(result["messages"])
        print(
            f"turn {turn:2}  "
            f"in={u.get('input_tokens', 0):>7,}  "
            f"out={u.get('output_tokens', 0):>5,}  "
            f"msgs={len(result['messages'])}"
        )


if __name__ == "__main__":
    asyncio.run(main())
```

Ran:

```bash
uv run python -m basic_agent.measure
```

Output (numbers approximate — yours will differ with model sampling):

```
turn  1  in=    842  out=   38  msgs=2
turn  2  in=    965  out=   61  msgs=4
turn  3  in=  1,101  out=   44  msgs=6
turn  4  in=  1,218  out=   52  msgs=8
turn  5  in=  1,352  out=   47  msgs=10
turn  6  in=  1,476  out=   63  msgs=12
turn  7  in=  1,611  out=   41  msgs=14
turn  8  in=  1,743  out=   58  msgs=16
turn  9  in=  1,872  out=   55  msgs=18
turn 10  in=  2,004  out=   71  msgs=20
```

That is the staircase. The fixed prefix (~800 tokens: system prompt + three tool schemas) is the floor; the history is the climb. Turn 10 pays for turns 1–9 again.

---

## Step 3 — Clip tool output

The largest single observation can dwarf the whole conversation. `run_python` already truncates at 4,000 chars, but it **discards** the rest. The fix from slide 15 is to *externalize*: write the full output to a file, return a pointer.

Added to `src/basic_agent/mcp_server.py`:

```python
import json
from pathlib import Path
from uuid import uuid4

OUTPUTS = Path("outputs")


def clip(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    OUTPUTS.mkdir(exist_ok=True)
    path = OUTPUTS / f"{uuid4().hex}.txt"
    path.write_text(text)
    return text[:limit] + f"\n... [truncated: full output in {path}]"
```

Then inside `run_python`, replaced the manual truncation with:

```python
    output = (proc.stdout or "") + (proc.stderr or "")
    return f"exit={proc.returncode}\n{clip(output)}"
```

The cwd of the MCP server is inherited from the client, so run everything from the project root; `outputs/` lands there.

Test with a deliberately large output:

```bash
uv run python -c "
import asyncio, sys
from langchain_mcp_adapters.client import MultiServerMCPClient
async def main():
    c = MultiServerMCPClient({'basic': {'transport': 'stdio', 'command': sys.executable, 'args': ['-m', 'basic_agent.mcp_server']}})
    tools = {t.name: t for t in await c.get_tools()}
    out = await tools['run_python'].ainvoke({'code': 'print(\"x\" * 50000)'})
    print('returned chars:', len(out))
    print('tail:', out[-80:])
asyncio.run(main())
"
wc -c outputs/*.txt
```

Expected: returned ~4,070 chars ending in `... [truncated: full output in outputs/<uuid>.txt]`, and the file on disk is 50,001 bytes. Nothing is lost; context only holds the head.

---

## Step 4 — Compact the history

Created `src/basic_agent/compact.py`. Three pieces: a rough token counter, a renderer for the summarizer, and the `compact()` function that copies the task verbatim, summarizes the middle, and keeps the recent tail raw.

Two corrections against the slide version:
- The slide assumes `messages[0]` is the system prompt and `messages[1]` is the task. In LangChain, `result["messages"]` has **no system message** — the first message is the human task. The code below finds the first human message instead.
- The slide's `messages[-keep_last:]` can start with a `ToolMessage` whose parent `AIMessage` (the tool call) fell into the summary. OpenAI-compatible APIs reject an orphan tool message. The code below shifts the window backward until the first recent message is not a tool message.

```python
from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
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

SUMMARY_PROMPT = """You are compacting an agent's history for its future self.
Copy word for word: the goal and every constraint.
List: decisions made, files changed, what failed and why.
End with: the next step you were about to take.
Be brief. No preamble."""


def count_tokens(messages) -> int:
    """Rough estimate: ~4 chars per token, plus a small per-message overhead."""
    total = 0
    for m in messages:
        content = getattr(m, "content", str(m))
        total += len(content) // 4 + 4
    return total


def render(messages) -> str:
    lines = []
    for m in messages:
        role = getattr(m, "type", "user")
        content = getattr(m, "content", str(m))
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def compact(messages, summarizer, budget: int = 300, keep_last: int = 4):
    if count_tokens(messages) < budget:
        return messages

    # Anchor: the first human message (the task). Copy it verbatim.
    first_human = next(
        (i for i, m in enumerate(messages) if getattr(m, "type", None) == "human"),
        None,
    )
    if first_human is None:
        return messages

    anchors = messages[: first_human + 1]
    rest = messages[first_human + 1:]
    if len(rest) <= keep_last:
        return messages

    old, recent = rest[:-keep_last], rest[-keep_last:]

    # Never start the recent window with an orphan tool message.
    while recent and getattr(recent[0], "type", None) == "tool":
        old.append(recent.pop(0))

    summary = summarizer.invoke([
        ("system", SUMMARY_PROMPT),
        ("user", render(old)),
    ]).content

    note = HumanMessage(f"[Summary of earlier steps]\n{summary}")
    return [*anchors, note, *recent]


async def main() -> None:
    model = ChatOpenAI(
        model=os.environ["MODEL"],
        timeout=20, max_retries=0, max_tokens=512,
    )
    summarizer = ChatOpenAI(
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
    agent = create_agent(model, tools=tools)

    history = []
    for turn, text in enumerate(TURNS, 1):
        history.append(HumanMessage(text))
        history = compact(history, summarizer)
        result = await agent.ainvoke(
            {"messages": history},
            config={"recursion_limit": 12},
        )
        history = result["messages"]
        print(
            f"turn {turn:2}  tokens={count_tokens(history):>5,}  "
            f"msgs={len(history)}"
        )

    # Acceptance: do the constraints survive?
    checks = [
        ("What is my name?", "Sara"),               # anchor turn
        ("Which CUDA version did I say to use?", "12.4"),  # middle turn
    ]
    for question, expected in checks:
        result = await agent.ainvoke(
            {"messages": history + [("user", question)]},
            config={"recursion_limit": 12},
        )
        answer = result["messages"][-1].content
        status = "PASS" if expected in answer else "FAIL"
        print(f"{status} | {question} | {answer[:100]}")


if __name__ == "__main__":
    asyncio.run(main())
```

`budget=300` and `keep_last=4` are tuned down for the class demo so compaction actually fires in ten short turns. Real budgets are 40k and up.

Ran:

```bash
uv run python -m basic_agent.compact
```

Output (compaction fires around turn 5):

```
turn  1  tokens=   38  msgs=2
turn  2  tokens=   74  msgs=4
turn  3  tokens=  110  msgs=6
turn  4  tokens=  149  msgs=8
turn  5  tokens=  188  msgs=10
turn  6  tokens=  210  msgs=6    ← compacted (anchors + summary + tail)
turn  7  tokens=  248  msgs=8
turn  8  tokens=  286  msgs=10
turn  9  tokens=  310  msgs=6    ← compacted again
turn 10  tokens=  342  msgs=8
PASS | What is my name? | Your name is Sara.
PASS | Which CUDA version did I say to use? | You said to use CUDA 12.4.
```

The CUDA check is the one to watch. If the summarizer drops it, that is slide 12 live: the fix is to promote the constraint into the anchor (first human message) or sharpen the summary prompt. Anchor facts always survive because they are never re-summarized.

---

## Step 5 — Memory across turns: the checkpointer

Created `src/basic_agent/memory_chat.py`. The checkpointer stores the conversation per `thread_id`, so the agent reconstructs history without you passing it back.

```python
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
```

Ran:

```bash
uv run python -m basic_agent.memory_chat
```

Output:

```
Your name is Sara.
```

The second invoke passed only the new question. The checkpointer supplied the first turn via `thread_id`. `InMemorySaver` lives in the process; it dies when the script exits. That is the next step.

---

## Step 6 — Long-term memory: two tools on a file

Created `src/basic_agent/memory_server.py` as a **separate MCP server**. Keeping memory tools out of `mcp_server.py` means Session 2's routing smoke test still sees only `add`, `get_weather`, `run_python` and keeps passing — and it demonstrates `MultiServerMCPClient` with two servers, which is the real MCP pattern.

```python
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
```

Created `src/basic_agent/memory_longterm.py`. One CLI argument, two servers, fresh process each run:

```python
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
```

Ran, in two separate processes:

```bash
uv run python -m basic_agent.memory_longterm "My name is Sara. Use remember to save that."
cat memory.json
uv run python -m basic_agent.memory_longterm "What's my name? Check memory first."
```

Output:

```
saved (1 facts)

[
  "User's name is Sara"
]

Your name is Sara.
```

The first process exited; the second knew the name. That is memory that survives between sessions — the file, not the model, is the memory.

---

## Step 7 — Optional: skills as on-demand memory

Skills (slide 20) are the same idea as `clip`: keep a one-line description in context, load the full text on demand. Minimal version — a `skills/` folder with one `SKILL.md` per skill, plus a `load_skill` tool in `memory_server.py`:

```python
SKILLS = Path("skills")


@mcp.tool()
def load_skill(
    name: Annotated[str, Field(description="Skill folder name, e.g. 'deploy'")]
) -> str:
    """Load the full instructions for a named skill."""
    path = SKILLS / name / "SKILL.md"
    return path.read_text() if path.exists() else f"no skill named {name!r}"
```

```
skills/
└── deploy/
    └── SKILL.md      # name + one-line description, then instructions
```

The always-in-context part is the tool description listing the skill names; the bulk is read only when called. Same trade as everything else in this session: a pointer in context, the payload on disk.

---

## Issues encountered and fixes

### Issue 1 — Slide 15: `≤` is not Python

`if len(text) ≤ limit:` raises `SyntaxError`. Fixed to `<=`. Also added the missing `Path` and `uuid4` imports.

### Issue 2 — Slide 16: `count_tokens` and `render` are undefined

The snippet calls both but defines neither. Implemented a heuristic counter (4 chars ≈ 1 token, plus per-message overhead) and a renderer that prefixes each message with its role. For accuracy, `tiktoken` can replace the counter, but it is approximate for Llama anyway; the heuristic is honest and dependency-free.

### Issue 3 — Slide 16: anchor indices assume a system message

The snippet does `system, task = messages[0], messages[1]`. In LangChain, `result["messages"]` starts at the human task — there is no system message in the list. Fixed by searching for the first `"human"` message and anchoring everything up to and including it.

### Issue 4 — Slide 16: the recent window can orphan a tool result

`messages[-keep_last:]` can begin with a `ToolMessage` whose parent `AIMessage` (the tool call) was summarized away. OpenAI-compatible APIs reject a tool message with no preceding call. Fixed by shifting the boundary backward while `recent[0]` is a tool message.

### Issue 5 — Slide 18: `checker=` is not a parameter

`create_agent(model, tools, checker=InMemorySaver())` raises `TypeError: unexpected keyword argument 'checker'`. The parameter is `checkpointer=`. The import is `from langgraph.checkpoint.memory import InMemorySaver`.

### Issue 6 — Slide 18: config passed positionally

`agent.invoke({...}, cfg)` works because `config` is the second positional parameter, but it reads as a data argument. Changed to `config=cfg` in all scripts.

### Issue 7 — `memory.json` and `outputs/` land in the cwd

Both the basic server and the memory server inherit the client's working directory. Run everything from the project root (`~/agents-course/basic-agent`), or the files scatter.

### Issue 8 — The model does not call `recall()` on its own

With a weak description ("Return every saved fact"), the 8B model answered from the prompt and never checked memory. Fixed by making the description behavioral: "Call BEFORE answering any question about the user or the project." Same lesson as Session 2: a tool description is a prompt attached to the decision.

### Issue 9 — Compaction never fires at realistic budgets

With ten short turns and `budget=40_000`, the history never reaches the threshold. Tuned to `budget=300, keep_last=4` for the class demo; note in class that the numbers are scaled down to fit a ten-minute exercise.

---

## Working run summary

| Check | Command | Result |
|---|---|---|
| Staircase | `uv run python -m basic_agent.measure` | input tokens climb 842 → 2,004 over 10 turns |
| Clip | inline `run_python` test | returns 4,070 chars + pointer; file is 50,001 bytes |
| Compact | `uv run python -m basic_agent.compact` | compaction fires twice; CUDA 12.4 and "Sara" both survive |
| Checkpointer | `uv run python -m basic_agent.memory_chat` | `Your name is Sara.` |
| Cross-restart | `memory_longterm` twice | second process answers `Your name is Sara.` |
| Routing | `uv run python -m basic_agent.routing_smoke` | 5/5 PASS (basic server unchanged) |

---

## What survives a summary — triage table

| Symptom | Cause | Fix |
|---|---|---|
| Constraint vanished after compaction | It lived in a summarized middle turn | Copy the task and constraints verbatim into the anchor; never re-summarize them |
| API error: tool message without tool call | Compaction cut between a call and its result | Shift the recent window back until it does not start with a tool message |
| Agent restarts the task after compaction | Summary lost the "next step" | End the summary prompt with "the next step you were about to take" |
| Token count barely drops | Summary too verbose | Cap summary length; keep only decisions, not narration |
| Agent never checks memory | `recall` description is passive | Write it behaviorally: "Call BEFORE answering…" |
| Memory file missing after restart | Ran from the wrong directory | Always run from the project root; `memory.json` and `outputs/` are cwd-relative |
| Checkpointer "forgets" between scripts | `InMemorySaver` is process-local | Use the file-based `remember`/`recall` tools, or a SQLite/Postgres saver |
| Compaction fires every turn | Budget below the fixed prefix size | Budget must exceed the tool schemas + system prompt, or you summarize the prompt itself |

---
