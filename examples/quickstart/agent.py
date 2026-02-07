"""Quickstart example — minimal working deep agent.

Usage:
    # Requires GOOGLE_API_KEY or Vertex AI configuration
    python examples/quickstart/agent.py
"""

from adk_deepagents import create_deep_agent

# Create a deep agent with default settings:
# - Gemini 2.5 Flash model
# - Filesystem tools (ls, read, write, edit, glob, grep)
# - Todo tools (write_todos, read_todos)
# - StateBackend (in-memory file storage)
root_agent = create_deep_agent(
    name="quickstart_agent",
    instruction="You are a helpful coding assistant. Use your filesystem and todo tools to help the user.",
)
