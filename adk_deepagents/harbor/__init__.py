"""Harbor integration for adk-deepagents.

Provides the HarborBackend (proxies deepagent file ops into the Harbor task
container), HarborAdapter (BaseAgent subclass for Harbor benchmarking), and
a Harbor-aware execution tool.

Install the extra to use:
    uv pip install "adk-deepagents[harbor]"
"""

from .adapter import HarborAdapter, to_atif
from .backend import HarborBackend
from .execution import create_harbor_execute_tool

__all__ = [
    "HarborAdapter",
    "HarborBackend",
    "create_harbor_execute_tool",
    "to_atif",
]
