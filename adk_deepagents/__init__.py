"""adk-deepagents — deepagents re-implemented with Google ADK primitives."""

__version__ = "0.1.0"

from adk_deepagents.graph import create_deep_agent, create_deep_agent_async
from adk_deepagents.types import (
    DynamicTaskConfig,
    SkillsConfig,
    SubAgentSpec,
    SummarizationConfig,
    TruncateArgsConfig,
)

__all__ = [
    "create_deep_agent",
    "create_deep_agent_async",
    "DynamicTaskConfig",
    "SkillsConfig",
    "SubAgentSpec",
    "SummarizationConfig",
    "TruncateArgsConfig",
]
