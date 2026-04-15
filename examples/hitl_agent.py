"""hitl_agent.py — Human-in-the-loop: agent pauses and asks the user.

HITL needs no special framework. It's just a blocking tool. When the agent
calls ask_human(), the loop blocks on input(), the human types, and the
agent continues with that answer as the tool result.

Run with:  uv run examples/hitl_agent.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nanoagent import tool, Tool, Agent, AgentConfig


@tool
def ask_human(question: str) -> str:
    """Ask the human operator a question and wait for their response.

    Use this when you need human judgment, approval, or information that
    cannot be determined from context alone.
    """
    print(f"\n[AGENT ASKS]: {question}")
    answer = input("[YOUR ANSWER]: ").strip()
    return answer or "(no answer provided)"


@tool
def present_options(options: str) -> str:
    """Present a set of options to the human and return their choice.

    Format options as a newline-separated list. The human will select one.
    """
    print(f"\n[AGENT PRESENTS OPTIONS]:\n{options}")
    choice = input("[YOUR CHOICE]: ").strip()
    return choice or "(no choice made)"


if __name__ == "__main__":
    config = AgentConfig(
        system=(
            "You are a helpful planning assistant. When you need human input — "
            "to confirm direction, clarify requirements, or get approval — use "
            "ask_human or present_options. Never proceed on assumptions that "
            "affect the human's preferences without asking first."
        ),
        sink=lambda e: None,
    )
    agent = Agent(tools=[ask_human, present_options], config=config)

    print("Starting human-in-the-loop agent. You'll be asked questions mid-task.\n")
    result = agent.run(
        "I want to plan a technical blog post. Research what topics interest me "
        "by asking me questions, then propose a title and outline and get my "
        "approval before finalizing it.",
        stream=True,
    )
    print("\n\n[FINAL OUTPUT]:")
    print(result)
