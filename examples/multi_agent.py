"""multi_agent.py — Two nanoagents calling each other as tools.

The key insight: another agent is just a tool. No orchestration framework
needed. Agent B is registered as a tool on Agent A.

Run with:  uv run examples/multi_agent.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nanoagent import tool, Tool, Agent, AgentConfig


# ── Researcher agent ───────────────────────────────────────────────────
# Has access to (mocked) search tools. In production, plug in real ones.

def _mock_search(topic: str) -> str:
    """Simulated search — replace with real web_search in production."""
    return (
        f"Research findings on '{topic}':\n"
        "- Tokio is the dominant async runtime for Rust (40M+ downloads/month)\n"
        "- async-std is an alternative with a std-like API surface\n"
        "- smol is a minimal async runtime, ~1500 lines\n"
        "- Monoio is io_uring-native, designed for high-throughput servers\n"
        "- The ecosystem standardized on Tokio after the 2022 async working group report\n"
        "- Embassy targets embedded/no_std environments without heap allocation\n"
    )

researcher_config = AgentConfig(
    system=(
        "You are a technical researcher. When asked to research a topic, "
        "use the search tool, then synthesize findings into clear bullet points. "
        "Be concise — your output goes to a writer who will expand it."
    ),
    sink=lambda e: None,
    max_tokens=1024,
)

# Inline tool for the researcher (avoids polluting the global registry)
search_tool = Tool(
    name="search",
    description="Search for technical information on a topic.",
    input_schema={
        "type": "object",
        "properties": {"topic": {"type": "string"}},
        "required": ["topic"],
    },
    fn=lambda topic: _mock_search(topic),
)
researcher = Agent(tools=[search_tool], config=researcher_config)


# ── Writer agent with researcher as a tool ─────────────────────────────

research_tool = Tool(
    name="research",
    description=(
        "Use the researcher agent to gather technical information on a topic. "
        "Returns structured findings ready for writing."
    ),
    input_schema={
        "type": "object",
        "properties": {"topic": {"type": "string"}},
        "required": ["topic"],
    },
    fn=lambda topic: researcher.run_fresh(f"Research this topic thoroughly: {topic}"),
)

writer_config = AgentConfig(
    system=(
        "You are a technical writer. Use the research tool to gather information, "
        "then write a clear, well-structured summary. Aim for 200-300 words."
    ),
    sink=lambda e: None,
)
writer = Agent(tools=[research_tool], config=writer_config)


# ── Run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Writer agent is researching and writing...\n")
    result = writer.run(
        "Write a summary of recent developments in Rust async runtimes, "
        "covering the major options and when to use each.",
        stream=True,
    )
    print()
