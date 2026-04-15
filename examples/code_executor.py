"""code_executor.py — Tool that runs Python in a subprocess sandbox.

The model writes Python, executes it via this tool, observes stdout/stderr,
and iterates. This is the foundation of code-writing agents.

Safety: subprocess isolation, 10-second timeout, no network access granted.

Run with:  uv run examples/code_executor.py
"""
import os
import subprocess
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nanoagent import tool, run, AgentConfig, Agent


@tool
def run_python(code: str) -> str:
    """Execute Python code in a sandboxed subprocess. Returns stdout + stderr.

    The code runs with a 10-second timeout. No internet access. No file I/O
    outside /tmp. Use this to test code, run computations, or verify logic.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
            # No additional sandboxing beyond subprocess isolation —
            # for production use seccomp/nsjail/gVisor
        )
        output = ""
        if result.stdout:
            output += f"stdout:\n{result.stdout}"
        if result.stderr:
            output += f"stderr:\n{result.stderr}"
        if result.returncode != 0:
            output += f"\nexited with code {result.returncode}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: code execution timed out after 10 seconds"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


if __name__ == "__main__":
    config = AgentConfig(
        system="You are a Python expert. Write code, run it with run_python, "
               "observe the output, and refine if needed. Show your work.",
        sink=lambda e: None,
    )
    agent = Agent(tools=[run_python], config=config)
    result = agent.run(
        "Write and run a Python script that finds all prime numbers under 100 "
        "using the Sieve of Eratosthenes. Print the count and the primes.",
        stream=True,
    )
    print()
