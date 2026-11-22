# Project 2 — Build Journal: MCP Tools, Sandboxed Code, and Routing Tests

Chronological record of extending the Project 1 ReAct agent with MCP tools and a sandboxed code execution tool, using LangChain MCP adapters, FastMCP, and Docker, running in WSL2 Ubuntu on a Surface Pro 7 (8 GB RAM).

- **Status:** ✅ Working
- **Date:** 2026-11-11
- **Files in repo:** `src/basic_agent/mcp_server.py`, `src/basic_agent/mcp_main.py`, `src/basic_agent/routing_smoke.py` (contents committed separately — this doc is the *process*)

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
| Docker | 27.5.1 (installed via apt inside WSL2) |

---

## Step 1 — Branch, Docker setup, and dependencies

### 1a. Branch and verify Project 1 still works

```bash
cd ~/agents-course/basic-agent
git checkout -b session2-tool-use
uv sync
uv run python -m basic_agent.main          # Project 1 still works?
```

### 1b. Install Docker (native apt method inside WSL2)

If Docker is not already installed, run this block. **Important:** `newgrp docker` starts a new shell. After it, you must `cd` back to the project directory before continuing.

```bash
sudo apt update
sudo apt install -y docker.io
sudo service docker start
sudo usermod -aG docker "$USER"
newgrp docker

# You are now in a new shell. Return to the project directory:
cd ~/agents-course/basic-agent
docker run --rm hello-world
```

> If `newgrp` is not available or you prefer a clean session, log out and back in (or run `exec su -l "$USER"`), then `cd ~/agents-course/basic-agent`.

### 1c. Add Session 2 dependencies

```bash
# Confirm you are in the project directory and Docker works
pwd                                        # should end with basic-agent
docker --version
docker pull python:3.12-slim               # one-time

# Add MCP dependencies
uv add mcp langchain-mcp-adapters
uv add "mcp[cli]"

# Verify imports
uv run python -c "from mcp.server.fastmcp import FastMCP; print('MCP OK')"
uv run python -c "from langchain_mcp_adapters.client import MultiServerMCPClient; print('Adapter OK')"
```

Both should print OK.

---

## Step 2 — Write the MCP server

Created `src/basic_agent/mcp_server.py`. Three tools: `add`, `get_weather`, `run_python`. Used FastMCP and `Annotated`/`Field` for schemas.

Key rules applied:
- No `print()` to stdout (stdio protocol). Use stderr for logs.
- Tool descriptions are routing instructions, not just documentation.
- `run_python` sandbox: no network, CPU/memory/PID limits, read-only FS, timeout, output truncation.

```python
from __future__ import annotations

import subprocess
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("basic-agent-tools")


@mcp.tool()
def add(
    a: Annotated[int, Field(description="First integer to add")],
    b: Annotated[int, Field(description="Second integer to add")],
) -> int:
    """Add two integers exactly. Use this for arithmetic instead of estimating."""
    return a + b


@mcp.tool()
def get_weather(
    city: Annotated[str, Field(description="City name, e.g. Paris")]
) -> str:
    """Get the current weather for a city. This is a stub for class use."""
    return "sunny, 22°C"


@mcp.tool()
def run_python(
    code: Annotated[str, Field(description="Short Python code to execute")]
) -> str:
    """Execute short Python code in an isolated Docker sandbox.

    Use ONLY for nontrivial computation or data transformation.
    Do NOT use for simple addition; use the add tool.
    The sandbox has no network, a 5-8 second timeout, and 128 MB memory.
    """
    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--memory", "128m",
        "--cpus", "0.5",
        "--pids-limit", "64",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=16m,noexec,nosuid",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        "python:3.12-slim",
        "python", "-I", "-c", code,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except FileNotFoundError:
        return "ERROR: docker not found. Sandbox unavailable."
    except subprocess.TimeoutExpired:
        return "ERROR: sandbox timed out after 8 seconds."

    output = (proc.stdout or "") + (proc.stderr or "")
    if len(output) > 4000:
        output = output[:4000] + "\n...[truncated]"

    return f"exit={proc.returncode}\n{output}"


if __name__ == "__main__":
    mcp.run()
```

Ran the MCP server once to confirm it starts:

```bash
uv run python -m basic_agent.mcp_server
```

It waits silently — normal for stdio transport. Press `Ctrl+C`.

---

## Step 3 — Connect LangChain to the MCP server

Created `src/basic_agent/mcp_main.py`. Uses `MultiServerMCPClient` to spawn the MCP server via stdio.

**Important addition:** added `max_tokens=512` to `ChatOpenAI` to avoid errors (long outputs / rate-limit issues on Groq free tier).

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
```

Ran from the project root:

```bash
cd ~/agents-course/basic-agent
uv run python -m basic_agent.mcp_main
```

The agent loaded three MCP tools and called `add`, `get_weather`, and `run_python`.

---

## Step 4 — Test the sandboxed code tool directly

Before trusting the agent, tested the sandbox manually.

Computation test:

```bash
docker run --rm --network none --memory 128m --cpus 0.5 \
  --pids-limit 64 --read-only --tmpfs /tmp:rw,size=16m,noexec,nosuid \
  --cap-drop ALL --security-opt no-new-privileges --user 65534:65534 \
  python:3.12-slim python -I -c "print(2**20)"
```

Network isolation test (should fail):

```bash
docker run --rm --network none python:3.12-slim \
  python -I -c "import urllib.request; urllib.request.urlopen('https://example.com')"
```

Network was blocked as expected. This is the point: **code as action must be sandboxed before the model is allowed to use it.**

---

## Step 5 — Write the routing smoke test

Created `src/basic_agent/routing_smoke.py`. Added `max_tokens=512` to the model.

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

CASES = [
    ("What is 47832 + 91347? Use the add tool.", {"add"}),
    ("What is the weather in Paris?", {"get_weather"}),
    ("Use add for 47 + 30, and also tell me the weather in Paris.", {"add", "get_weather"}),
    ("Use run_python to compute 2**20.", {"run_python"}),
    ("Tell me a joke.", set()),
]


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
    agent = create_agent(model, tools=tools)

    for query, expected in CASES:
        result = await agent.ainvoke(
            {"messages": [("user", query)]},
            config={"recursion_limit": 12},
        )

        called = {
            tc["name"]
            for m in result["messages"]
            for tc in (getattr(m, "tool_calls", None) or [])
        }

        status = "PASS" if called == expected else "FAIL"
        print(f"{status} | {query[:55]:55} | expected={expected} got={called}")


if __name__ == "__main__":
    asyncio.run(main())
```

Ran:

```bash
uv run python -m basic_agent.routing_smoke
```

All five passed.

---

## Step 6 — Issues encountered and fixes

### Issue 1 — Model errors without `max_tokens`

Initially omitted `max_tokens` from `ChatOpenAI`. Some runs produced very long outputs or hit Groq free-tier rate limits, causing errors. Setting `max_tokens=512` resolved this. Applied consistently in `mcp_main.py` and `routing_smoke.py`.

### Issue 2 — MCP server stdout corruption

Accidentally left a `print()` in `mcp_server.py` during early testing. That corrupted the stdio protocol and the client hung. Removed all stdout prints; used `logging` to stderr if needed.

### Issue 3 — Docker permission denied in WSL2

Docker commands failed from Ubuntu at first. Fixed by adding the user to the `docker` group and refreshing the session:

```bash
sudo usermod -aG docker "$USER"
newgrp docker
cd ~/agents-course/basic-agent   # new shell resets context
```

Then verified with `docker run --rm hello-world`.

### Issue 4 — Tool overlap: `add` vs `run_python`

The model sometimes used `run_python` for simple addition. Fixed by strengthening descriptions:
- `add`: "Add two integers exactly. Use this for arithmetic instead of estimating."
- `run_python`: "Use ONLY for nontrivial computation... Do NOT use for simple addition; use the add tool."

### Issue 5 — `12+30` not calling `add`

Same as Project 1: small models do trivial arithmetic in-head. Use big numbers or explicit "use the add tool" phrasing.

### Issue 6 — MCP tools not loading

`MultiServerMCPClient` initially used `command="python"`, which resolved to system Python 3.14 and missed the project venv. Fixed to `command=sys.executable`, ensuring the MCP server runs inside the project environment.

---

## Step 7 — Working run

`uv run python -m basic_agent.mcp_main` output:

```
Loaded MCP tools: ['add', 'get_weather', 'run_python']
[human] Use the add tool for 47832 + 91347, tell me the weather in Paris, and use run_python to compute 2**20.
[ai] 
   -> add({'a': 47832, 'b': 91347})
   -> get_weather({'city': 'Paris'})
   -> run_python({'code': 'print(2**20)'})
[tool] 139179
[tool] sunny, 22°C
[tool] exit=0
1048576
[ai] 47832 + 91347 = 139179. Paris is sunny at 22°C. 2**20 = 1048576.
```

Routing smoke test output:

```
PASS | What is 47832 + 91347? Use the add tool.               | expected={'add'} got={'add'}
PASS | What is the weather in Paris?                          | expected={'get_weather'} got={'get_weather'}
PASS | Use add for 47 + 30, and also tell me the weather...   | expected={'add', 'get_weather'} got={'add', 'get_weather'}
PASS | Use run_python to compute 2**20.                       | expected={'run_python'} got={'run_python'}
PASS | Tell me a joke.                                        | expected=set() got=set()
```

---

## Wrong-tool triage table

Collected real student cases from homework and filled in this table.

| Symptom | Likely cause | Fix |
|---|---|---|
| Model does `12+30` in-head, no `add` call | Small model shortcut | Use large numbers or say “use the add tool” |
| Model calls `run_python` for `47+30` | Tool overlap | Strengthen `run_python` description: “Do NOT use for simple addition; use add” |
| Model calls `add` for `2**20` | `add` description too broad | Say `add` is only for two integer operands |
| Model calls no tool for weather | Tool description vague | Add “current weather for a city” and an example city |
| Model calls both tools for a single-intent query | Ambiguous prompt | Rewrite query with explicit tool names |
| Model invents arguments | Schema unclear | Add `Annotated` + `Field` descriptions; FastMCP validation returns an error |
| Model never calls tools in parallel | Model limitation or prompt | Explicitly say “call both tools in the same turn”; harness already executes all `tool_calls` |
| Model calls a tool not exposed | Tool filtering issue | Check `client.get_tools()` output; expose only relevant MCP servers |
| `run_python` returns Docker error | Docker not running / image missing | `docker pull python:3.12-slim`; check Docker service |
| MCP server hangs or breaks | `print()` to stdout | Remove stdout prints; log to stderr |

