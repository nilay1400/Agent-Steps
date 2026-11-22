# Agents Course 

Hands-on projects for the AI Agents course. Each session has a build journal.

## Sessions

| Session | Topic | Build journal | Status |
|---|---|---|---|
| 1 | What is an agent? Definition, the loop, tool calls are just text, the harness | [Project 1 — Basic Agent with Two Tools](docs/project-1-basic-agent.md) | ✅ Working |
| 2 | Tool use: schemas, parallel calls, code as action, constrained decoding, MCP | [Project 2 — MCP Tools, Sandboxed Code, and Routing Tests](docs/project-2-mcp-tools.md) | ✅ Working |

---
## Repository layout

```
agents-course/
└── basic-agent/                 # the project (uv package)
    ├── src/basic_agent/
    │   ├── tools.py             # S1: add, get_weather
    │   ├── main.py              # S1: LangChain create_agent
    │   ├── mcp_server.py        # S2: FastMCP server (add, get_weather, run_python)
    │   ├── mcp_main.py          # S2: agent over MultiServerMCPClient
    │   └── routing_smoke.py     # S2: tool-routing smoke test
    ├── docs/
    │   ├── project-1-basic-agent.md
    │   └── project-2-mcp-tools.md
    ├── pyproject.toml
    └── .env                     # never committed
```

---

## Quick start

```bash
cd basic-agent
uv sync
# create .env (see Project 1 journal)
uv run python -m basic_agent.main          # Session 1 agent
uv run python -m basic_agent.mcp_main      # Session 2 agent (MCP + sandbox)
uv run python -m basic_agent.routing_smoke # Session 2 routing tests
```

Prerequisites: `uv`, Docker (for the Session 2 sandbox), and a Groq API key (free tier, OpenAI-compatible).
