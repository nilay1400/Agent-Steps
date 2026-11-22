import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent          # NEW, non-deprecated import
from .tools import add, get_weather

# Load .env explicitly — walks up to ~/agents-course/.env
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

llm = ChatOpenAI(
    model=os.getenv("MODEL", "llama-3.1-8b-instant"),
    temperature=0,
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30,        # fail fast instead of hanging
    max_retries=0,     # surface 429s, don't silently back off
    max_tokens=512,
)

agent = create_agent(llm, [add, get_weather])

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [("user", "What's 12+30 and the weather in Paris?")]},
        config={"recursion_limit": 10},
    )
    for msg in result["messages"]:
        print(f"[{msg.type}] {msg.content}")
        for tc in getattr(msg, "tool_calls", None) or []:
            print(f"   -> {tc['name']}({tc['args']})")
