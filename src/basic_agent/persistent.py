import socket
_orig = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **kw: [r for r in _orig(*a, **kw) if r[0] == socket.AF_INET]

import json
import os
import re
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import RemoveMessage

from .tools import add, get_weather

# .env discovery
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / ".env").exists():
        load_dotenv(parent / ".env")
        break

llm = ChatOpenAI(
    model=os.getenv("MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
    temperature=0,
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30,
    max_retries=2,
)

# SQLite checkpointer — survives process restarts
DB_PATH = Path(__file__).parent.parent.parent / "memory.db"
conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
checkpointer = SqliteSaver(conn)

# ---------------------------------------------------------------------------
# Long-term fact store — MUST be before create_agent
# ---------------------------------------------------------------------------
FACTS_FILE = Path(__file__).parent.parent.parent / "facts.json"

def load_facts() -> dict:
    if FACTS_FILE.exists():
        return json.loads(FACTS_FILE.read_text())
    return {}

def save_facts(facts: dict) -> None:
    FACTS_FILE.write_text(json.dumps(facts, indent=2))

@tool
def remember(key: str, value: str) -> str:
    """Save a durable fact about the user (name, preferences, locations).
    Use only for facts that should persist across conversations."""
    facts = load_facts()
    facts[key] = value
    save_facts(facts)
    return f"remembered {key}={value}"

# ---------------------------------------------------------------------------
# Retry wrapper for Groq's transient "Parsing failed" 400s
# ---------------------------------------------------------------------------
def _invoke_with_retry(agent, payload, config, attempts: int = 3):
    last = None
    for i in range(attempts):
        try:
            return agent.invoke(payload, config)
        except Exception as e:
            msg = str(e)
            transient = (
                "Parsing failed" in msg
                or "could not be parsed" in msg
                or "Error code: 400" in msg
                or "Error code: 500" in msg
                or "Error code: 502" in msg
                or "Error code: 503" in msg
            )
            last = e
            if not transient:
                raise
            time.sleep(0.5 * (i + 1))
    raise last

# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------
MAX_MESSAGES = 20

def compact_if_needed(thread_id: str) -> None:
    """If the thread has more than MAX_MESSAGES, summarize older messages.
    Sanitizes the summary to plain text before injecting. Never raises."""
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.get_state(config)
    messages = state.values.get("messages", [])

    if len(messages) <= MAX_MESSAGES:
        return

    split = len(messages) // 2
    old, recent = messages[:split], messages[split:]

    transcript_lines = []
    for m in old:
        text = m.content if isinstance(m.content, str) else str(m.content)
        if text.strip():
            transcript_lines.append(f"[{m.type}] {text}")

    if not transcript_lines:
        return

    try:
        summary = llm.invoke(
            [
                ("system",
                 "Summarize the conversation in 2-3 plain sentences. "
                 "No bullets, no markdown, no lists, no quotes. "
                 "Preserve facts, names, locations, preferences."),
                ("user", "\n".join(transcript_lines)),
            ]
        ).content
        summary = re.sub(r"[*_`#>]", "", summary).strip()
    except Exception as e:
        print(f"[compaction skipped: {type(e).__name__}: {str(e)[:120]}]")
        return

    removals = [RemoveMessage(id=m.id) for m in old]
    agent.update_state(config, {"messages": removals})
    agent.update_state(
        config,
        {"messages": [("system", f"Prior context summary: {summary}")]},
    )
    print(f"[compacted {len(old)} messages into summary]")

# ---------------------------------------------------------------------------
# Agents — remember is now in scope
# ---------------------------------------------------------------------------
agent = create_agent(llm, [add, get_weather, remember], checkpointer=checkpointer)
agent_plain = create_agent(llm, [], checkpointer=checkpointer)

def chat(user_msg: str, thread_id: str) -> str:
    compact_if_needed(thread_id)

    facts = load_facts()
    messages = []
    if facts:
        facts_str = "\n".join(f"- {k}: {v}" for k, v in facts.items())
        messages.append(("system", f"Known durable facts about the user:\n{facts_str}"))
    messages.append(("user", user_msg))

    config = {"configurable": {"thread_id": thread_id}}
    r = _invoke_with_retry(agent, {"messages": messages}, config)
    return r["messages"][-1].content

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python -m basic_agent.persistent <thread_id> <message>")
        sys.exit(1)
    thread_id, message = sys.argv[1], " ".join(sys.argv[2:])
    print(chat(message, thread_id))
