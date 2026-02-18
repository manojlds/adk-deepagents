# Structured Output

## Overview

adk-deepagents supports constraining agent output with a Pydantic `BaseModel`. When an `output_schema` is provided, the agent's responses are validated against the schema, ensuring structured and predictable output.

## How It Works

The `output_schema` parameter is passed directly to ADK's `LlmAgent`:

```python
agent = LlmAgent(
    ...,
    output_schema=output_schema,
)
```

ADK handles the schema enforcement with the underlying model, instructing it to return JSON that conforms to the Pydantic model.

## When to Use

- **Data extraction** — Pull structured data from unstructured text
- **Structured analysis** — Return analysis results in a consistent format
- **API responses** — Generate responses that match an API contract
- **Classification** — Categorize input into predefined categories
- **Multi-field responses** — When you need multiple distinct output fields

## Examples

### Basic: AnalysisResult Model

```python
from pydantic import BaseModel

from adk_deepagents import create_deep_agent


class AnalysisResult(BaseModel):
    summary: str
    sentiment: str
    confidence: float
    key_topics: list[str]


agent = create_deep_agent(
    instruction="Analyze the given text and return a structured analysis.",
    output_schema=AnalysisResult,
)
```

The agent will return responses conforming to:

```json
{
    "summary": "The article discusses...",
    "sentiment": "positive",
    "confidence": 0.92,
    "key_topics": ["AI", "machine learning", "ethics"]
}
```

### Complex Nested Models

```python
from pydantic import BaseModel

from adk_deepagents import create_deep_agent


class CodeIssue(BaseModel):
    file: str
    line: int
    severity: str
    description: str
    suggestion: str


class CodeReviewResult(BaseModel):
    overall_quality: str
    score: int
    issues: list[CodeIssue]
    strengths: list[str]
    recommendations: list[str]


agent = create_deep_agent(
    instruction=(
        "Review the provided code and return a structured code review. "
        "Score from 1-10. Severity is one of: critical, warning, info."
    ),
    output_schema=CodeReviewResult,
)
```

### With Tools and Structured Output

```python
from pydantic import BaseModel

from adk_deepagents import create_deep_agent


class ProjectSummary(BaseModel):
    name: str
    language: str
    file_count: int
    total_lines: int
    description: str


agent = create_deep_agent(
    instruction=(
        "Explore the project filesystem and return a structured summary. "
        "Use ls, read_file, and glob to gather information."
    ),
    output_schema=ProjectSummary,
)
```

The agent will use its filesystem tools to explore the project, then return a structured response matching the `ProjectSummary` schema.
