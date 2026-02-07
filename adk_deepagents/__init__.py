"""adk-deepagents — deepagents re-implemented with Google ADK primitives."""

__version__ = "0.1.0"

from adk_deepagents.graph import create_deep_agent, create_deep_agent_async
from adk_deepagents.types import SkillsConfig, SubAgentSpec, SummarizationConfig

__all__ = [
    "create_deep_agent",
    "create_deep_agent_async",
    "SkillsConfig",
    "SubAgentSpec",
    "SummarizationConfig",
]
