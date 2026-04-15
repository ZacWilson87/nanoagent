"""web_researcher.py — Multi-tool: search + fetch + summarize.

Uses real APIs when environment variables are present; falls back to fixture
data so the example is always runnable without credentials.

Run with:  uv run examples/web_researcher.py
Optional env vars:
  BRAVE_API_KEY   — Brave Search API key for real search results
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import httpx
from nanoagent import tool, run, AgentConfig, Agent

# ── Fixture data (used when no search API key is present) ──────────────

_FIXTURE_RESULTS = """
Search results for 'AI agent frameworks comparison 2024':
1. LangChain — Popular Python framework with 80k+ GitHub stars. Abstracts over
   LLM calls with chains, agents, and tools. Steep learning curve.
2. LangGraph — Graph-based orchestration built on LangChain. Useful for
   complex multi-step workflows with cycles.
3. CrewAI — Multi-agent collaboration framework. Role-based agents with
   manager/worker patterns.
4. AutoGen — Microsoft's framework for conversational multi-agent systems.
5. nanoagent — Minimal ReAct loop in ≤300 lines. Educational reference
   implementation with no abstractions.
""".strip()

_FIXTURE_PAGE = """
AI agent frameworks have proliferated in 2024. The core pattern — ReAct
(Reason + Act) — is simple: model thinks, calls a tool, observes result,
repeats. Most frameworks add orchestration, persistence, and multi-agent
coordination on top. The tradeoff is abstraction vs. visibility. Minimal
implementations like nanoagent make the underlying algorithm clear.
""".strip()

# ── Tool definitions ───────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the web for information. Returns a list of relevant results."""
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        # Graceful degradation: return fixture data when no API key
        return f"[FIXTURE DATA — set BRAVE_API_KEY for real results]\n{_FIXTURE_RESULTS}"
    try:
        resp = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
            params={"q": query, "count": 5},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("web", {}).get("results", [])
        return "\n".join(f"{i+1}. {r['title']}: {r['description']}" for i, r in enumerate(results))
    except Exception as e:
        return f"Search failed: {e}\n{_FIXTURE_RESULTS}"


@tool
def fetch_page(url: str) -> str:
    """Fetch and return the text content of a web page (first 2000 chars)."""
    try:
        resp = httpx.get(url, timeout=10, follow_redirects=True)
        resp.raise_for_status()
        # Very rough HTML strip — good enough for demonstration
        text = resp.text
        for tag in ["<script", "<style", "<head"]:
            while tag in text:
                start = text.find(tag)
                end = text.find(">", start) + 1
                text = text[:start] + text[end:]
        import re
        text = re.sub(r"<[^>]+>", " ", text)
        text = " ".join(text.split())
        return text[:2000]
    except Exception:
        return _FIXTURE_PAGE


@tool
def summarize_text(text: str) -> str:
    """Summarize the key points from a block of text in 3-5 bullet points."""
    # This tool intentionally delegates summarization to the agent itself
    # by just returning the text — the model does the summarization.
    # In production you might call another model or use NLP here.
    return f"Text to summarize:\n{text[:1500]}"


# ── Run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = AgentConfig(
        system="You are a research assistant. Use web_search to find information, "
               "fetch_page for details, and synthesize your findings clearly.",
        sink=lambda e: None,  # suppress internal events for clean output
    )
    agent = Agent(tools=[web_search, fetch_page, summarize_text], config=config)
    result = agent.run(
        "Research the current state of AI agent frameworks and give me a "
        "brief comparison of the major options.",
        stream=True,
    )
    print()  # newline after streamed output
