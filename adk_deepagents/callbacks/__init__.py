"""Callback functions that replace the deepagents middleware stack."""

from adk_deepagents.callbacks.after_tool import make_after_tool_callback
from adk_deepagents.callbacks.before_agent import make_before_agent_callback
from adk_deepagents.callbacks.before_model import make_before_model_callback
from adk_deepagents.callbacks.before_tool import make_before_tool_callback

__all__ = [
    "make_after_tool_callback",
    "make_before_agent_callback",
    "make_before_model_callback",
    "make_before_tool_callback",
]
