"""adk-deepagents — deepagents re-implemented with Google ADK primitives."""

__version__ = "0.3.0"

from adk_deepagents.graph import create_deep_agent, create_deep_agent_async
from adk_deepagents.types import (
    BrowserConfig,
    CallbackHooks,
    DeepAgentConfig,
    DynamicTaskConfig,
    SkillsConfig,
    SubAgentSpec,
    SummarizationConfig,
    TemporalTaskConfig,
    TruncateArgsConfig,
)

__all__ = [
    "create_deep_agent",
    "create_deep_agent_async",
    "BrowserConfig",
    "CallbackHooks",
    "DeepAgentConfig",
    "DynamicTaskConfig",
    "SkillsConfig",
    "SubAgentSpec",
    "SummarizationConfig",
    "TemporalTaskConfig",
    "TruncateArgsConfig",
]
