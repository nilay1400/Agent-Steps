from __future__ import annotations

import asyncio
import os
import sys
import time
from collections import deque

import openai
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
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

COMPACT_PROMPT = """You are compacting an agent's history for its future self.

Output EXACTLY two sections, in this order, with these exact headers:

CONSTRAINTS:
- A constraint is a rule the user gave the agent to OBEY: a tool/library
  version, a hard requirement, a prohibition. Copy it verbatim, one per line,
  each starting with "- ".
- A fact about the user's system ("runs on a 4-core VM", "reads CSV",
  "bottleneck is parsing") is NOT a constraint. Do not list facts here.
- If there are no constraints, output exactly: - none

SUMMARY:
- Brief, no preamble. Include the goal, decisions made, what failed and why.
- End with the next step you were about to take.
"""

SUMMARY_MARKER = "[Summary of earlier steps]"
LEDGER_HEADER = "[Pinned constraints from earlier turns — treat as authoritative]"

# Override to skip preflight. If unset, resolve_model() picks from what's
# actually available in your Groq account.
AGENT_MODEL = os.environ.get("AGENT_MODEL")

# Preference: gpt-oss-120b first (reasoning_effort IS supported on Groq for
# these), then 20b, then any other chat model, then allam as last resort.
# Prompt-guard / whisper / orpheus are excluded by _is_chat_model().
PREFERRED_ORDER = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b",
    "allam-2-7b",
]

# Models we know don't accept reasoning_effort on Groq.
NO_EFFORT_SUPPORT = ("qwen3", "qwen/qwen3")

# Models that are not chat-completion endpoints.
NON_CHAT_PREFIXES = ("whisper", "canopylabs/", "meta-llama/llama-prompt-guard")


def _is_chat_model(mid: str) -> bool:
    return not mid.startswith(NON_CHAT_PREFIXES)


def _is_reasoning(mid: str) -> bool:
    m = mid.lower()
    if any(h in m for h in ("qwen3", "qwq", "deepseek-r1", "thinking")):
        return True
    if "gpt-oss" in m:
        return True
    return False


def _supports_effort(mid: str) -> bool:
    return not any(h in mid.lower() for h in NO_EFFORT_SUPPORT)


# OTPM below is what you *know* your tier gives you. If unsure, run one turn
# and read the 429 message: "Limit X, Used Y, Requested Z". Set OTPM=X.
OTPM = 1000
WINDOW_S = 60.0
SAFETY = 0.85

# Reasoning models must fit thinking + answer. 1024 covers ~80 reasoning +
# ~400 answer comfortably. Non-reasoning models only need the answer budget.
MAX_OUT = 1024 if True else 512     # see _build_model
MAX_OUT_SUMMARY = 768
VERBOSE = os.environ.get("COMPACT_VERBOSE", "0") == "1"


# ---------- model resolution ----------

async def resolve_model(preferred: str | None) -> tuple[str, bool, bool]:
    """Return (model_id, is_reasoning, supports_effort)."""
    client = openai.AsyncOpenAI()
    try:
        listing = await client.models.list()
        ids = [m.id for m in listing.data]
    except Exception as e:
        if preferred:
            print(f"  [model] could not list ({e}); using AGENT_MODEL={preferred!r}")
            return preferred, _is_reasoning(preferred), _supports_effort(preferred)
        raise RuntimeError(
            f"Could not list models ({e}) and AGENT_MODEL is not set."
        ) from e

    chat = [i for i in ids if _is_chat_model(i)]

    if preferred:
        if preferred in ids:
            print(f"  [model] using AGENT_MODEL={preferred!r}")
            return preferred, _is_reasoning(preferred), _supports_effort(preferred)
        print(f"  [model] AGENT_MODEL={preferred!r} NOT available; "
              f"chat models = {chat}")

    for cand in PREFERRED_ORDER:
        if cand in chat:
            print(f"  [model] auto-selected {cand!r}  "
                  f"(reasoning={_is_reasoning(cand)}, "
                  f"effort_supported={_supports_effort(cand)})")
            return cand, _is_reasoning(cand), _supports_effort(cand)

    if chat:
        pick = chat[0]
        print(f"  [model] no preferred model; using {pick!r} "
              f"(reasoning={_is_reasoning(pick)})")
        return pick, _is_reasoning(pick), _supports_effort(pick)

    raise RuntimeError(f"No chat models available. Full list: {ids}")


def _build_model(mid: str, max_tokens: int, reasoning: bool,
                 effort_ok: bool) -> ChatOpenAI:
    base = dict(model=mid, timeout=90, max_retries=0)
    if reasoning and effort_ok:
        # GPT-OSS models on Groq honor reasoning_effort. "low" -> ~30-80
        # reasoning tokens. max_tokens must cover both thinking + answer.
        print(f"  [model] {mid}: reasoning_effort='low', max_tokens={max_tokens}")
        return ChatOpenAI(
            **base, max_tokens=max_tokens,
            reasoning_effort="low",
            extra_body={"reasoning_effort": "low", "include_reasoning": False},
        )
    if reasoning and not effort_ok:
        # Qwen3 etc — effort not honored. Just give it a large budget.
        print(f"  [model] {mid}: effort unsupported; max_tokens={max_tokens}")
        return ChatOpenAI(
            **base, max_tokens=max(max_tokens, 2048),
            extra_body={"enable_thinking": False},
        )
    return ChatOpenAI(**base, max_tokens=max_tokens)


# ---------- pacing ----------

class Pacer:
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
            print(f"  [pace] {used}/{self.budget} output tokens; sleeping {wait:.1f}s")
            await asyncio.sleep(wait)

    def record(self, tokens: int) -> None:
        if tokens > 0:
            self.events.append((time.monotonic(), tokens))


async def call_with_retry(coro_fn, *args, attempts: int = 6, **kwargs):
    delay = 5.0
    for i in range(attempts):
        try:
            return await coro_fn(*args, **kwargs)
        except openai.RateLimitError:
            if i == attempts - 1:
                raise
            print(f"  [429] retrying in {delay:.0f}s")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)


# ---------- helpers ----------

def count_tokens(messages) -> int:
    total = 0
    for m in messages:
        content = getattr(m, "content", str(m))
        if not isinstance(content, str):
            content = str(content)
        total += len(content) // 4 + 4
    return total


def render(messages) -> str:
    lines = []
    for m in messages:
        role = getattr(m, "type", "user")
        content = getattr(m, "content", str(m))
        if not isinstance(content, str):
            content = str(content)
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def usage_out(resp) -> int:
    u = getattr(resp, "usage_metadata", None) or {}
    return int(u.get("output_tokens", 0))


def _is_prior_summary(m) -> bool:
    if getattr(m, "type", None) != "human":
        return False
    content = getattr(m, "content", "")
    return isinstance(content, str) and SUMMARY_MARKER in content


def parse_compact_output(text: str) -> tuple[list[str], str]:
    constraints: list[str] = []
    summary_lines: list[str] = []
    section: str | None = None
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("CONSTRAINTS:"):
            section = "c"; continue
        if s.upper().startswith("SUMMARY:"):
            section = "s"; continue
        if section == "c":
            cleaned = s.lstrip("-•* ").strip()
            if cleaned and cleaned.lower() != "none":
                constraints.append(cleaned)
        elif section == "s":
            summary_lines.append(line)
    return constraints, "\n".join(summary_lines).strip()


def _constraint_already_present(c: str, ledger: list[str]) -> bool:
    cl = c.lower().strip()
    return any(cl == el.lower().strip()
               or cl in el.lower() or el.lower() in cl
               for el in ledger)


def build_ledger_message(ledger: list[str]) -> SystemMessage:
    body = "\n".join(f"- {c}" for c in ledger) if ledger else "- none"
    return SystemMessage(f"{LEDGER_HEADER}\n{body}")


# ---------- compaction ----------

async def compact(messages, summarizer, pacer, ledger: list[str],
                  budget: int = 300, keep_last: int = 4):
    if count_tokens(messages) < budget:
        return messages

    first_human = next(
        (i for i, m in enumerate(messages) if getattr(m, "type", None) == "human"),
        None,
    )
    if first_human is None:
        return messages

    anchors = messages[: first_human + 1]
    rest = messages[first_human + 1:]

    prior_summary = None
    if rest and _is_prior_summary(rest[0]):
        prior_summary = rest[0]
        rest = rest[1:]

    if len(rest) <= keep_last:
        return messages

    old, recent = rest[:-keep_last], rest[-keep_last:]
    while recent and getattr(recent[0], "type", None) == "tool":
        old.append(recent.pop(0))

    if not old:
        return messages

    await pacer.acquire()
    resp = await call_with_retry(
        summarizer.ainvoke,
        [("system", COMPACT_PROMPT), ("user", render(old))],
    )
    pacer.record(usage_out(resp))

    new_constraints, summary = parse_compact_output(resp.content)

    added = []
    for c in new_constraints:
        if not _constraint_already_present(c, ledger):
            ledger.append(c); added.append(c)

    if VERBOSE:
        print("  [compact] raw summary:\n" + (resp.content or "(EMPTY)") + "\n  ---")
        print("  [compact] new constraints:", added or "none")
        print("  [compact] ledger now:", ledger)

    if prior_summary is not None:
        note = HumanMessage(f"{prior_summary.content}\n\n[More recent steps]\n{summary}")
    else:
        note = HumanMessage(f"{SUMMARY_MARKER}\n{summary}")

    return [*anchors, note, *recent]


# ---------- main ----------

def _diagnose_message(m) -> str:
    meta = getattr(m, "response_metadata", {}) or {}
    extra = getattr(m, "additional_kwargs", {}) or {}
    usage = getattr(m, "usage_metadata", None) or {}
    parts = [
        f"finish={meta.get('finish_reason')!r}",
        f"content_len={len(m.content or '')}",
        f"output_tokens={usage.get('output_tokens', 0)}",
    ]
    for name, src in (("meta", meta), ("extra", extra)):
        for key in ("reasoning_content", "reasoning", "thinking"):
            v = src.get(key)
            if v:
                parts.append(f"{name}.{key}_chars={len(str(v))}")
    return "  ".join(parts)


async def main() -> None:
    model_id, reasoning, effort_ok = await resolve_model(AGENT_MODEL)
    print(f"agent_model={model_id}  reasoning={reasoning}  effort_ok={effort_ok}")

    model = _build_model(model_id, MAX_OUT, reasoning, effort_ok)
    summarizer = _build_model(model_id, MAX_OUT_SUMMARY, reasoning, effort_ok)

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
    history: list = []
    ledger: list[str] = []

    for turn, text in enumerate(TURNS, 1):
        history.append(HumanMessage(text))
        history = await compact(history, summarizer, pacer, ledger)

        call_messages = [build_ledger_message(ledger), *history]
        await pacer.acquire()
        result = await call_with_retry(
            agent.ainvoke,
            {"messages": call_messages},
            config={"recursion_limit": 12},
        )
        history = [m for m in result["messages"]
                   if not isinstance(m, SystemMessage)]

        u = getattr(history[-1], "usage_metadata", None) or {}
        pacer.record(int(u.get("output_tokens", 0)))

        print(
            f"turn {turn:2}  tokens={count_tokens(history):>5,}  "
            f"msgs={len(history)}  constraints={len(ledger)}"
        )

    for question, expected in [
        ("What is my name?", "Sara"),
        ("Which CUDA version did I say to use?", "12.4"),
    ]:
        call_messages = [build_ledger_message(ledger), *history, ("user", question)]
        await pacer.acquire()
        result = await call_with_retry(
            agent.ainvoke,
            {"messages": call_messages},
            config={"recursion_limit": 12},
        )
        last = result["messages"][-1]
        u = getattr(last, "usage_metadata", None) or {}
        pacer.record(int(u.get("output_tokens", 0)))
        answer = last.content if isinstance(last.content, str) else str(last.content)
        print(f"  diag: {_diagnose_message(last)}")
        status = "PASS" if expected in answer else "FAIL"
        print(f"{status} | {question} | {answer[:200]!r}")


if __name__ == "__main__":
    asyncio.run(main())
