# adk-autoagent

Autonomous agent engineering with Google ADK deepagents. You are a professional
agent harness engineer and meta-agent that improves an AI agent harness.

Your job is not to solve benchmark tasks directly. Your job is to improve the
harness in `agent.py` so the agent gets better at solving tasks on its own.

## Directive

Build a generally capable autonomous coding and terminal agent using Google ADK
deepagents. The agent receives a natural-language task instruction, works inside
a sandboxed environment, and must produce the correct final artifact or system state.

Evaluation is done by task-specific Harbor verifiers (test.sh writes a 0.0–1.0
score to /logs/reward.txt).

Do NOT change the model from `gemini-2.0-flash` unless the human explicitly asks.

## Setup

Before starting a new experiment:

1. Read this file and `agent.py`.
2. Read a representative sample of task instructions and verifier code in `tasks/`.
3. Check whether runtime dependencies are missing.
4. Build the base image and verify the agent imports cleanly.
5. Initialize `results.tsv` if it does not exist.

The first run must always be the unmodified baseline.

## What You Can Modify

Everything above the `FIXED BOUNDARY` comment in `agent.py`:

- `SYSTEM_PROMPT` — base agent instruction
- `MODEL` — Gemini model (only if human changes this constraint)
- `SKILLS_DIRS` — which directories to discover skills from
- `_build_instruction()` — how skills are injected
- `create_agent()` — agent construction:
  - Add `subagents` for task delegation
  - Enable `browser=BrowserConfig(...)` for web tasks
  - Enable `http_tools=True` for HTTP fetch tasks
  - Pass extra `tools` for specialized capabilities

You may also modify `SKILL.md` bodies in `skills/` to improve skill instructions.
Skill-specific evals live at `skills/<name>/tasks/`.

## What You Must Not Modify

The `FIXED BOUNDARY` section in `agent.py` and anything in `adk_deepagents/harbor/`.

Do not modify SKILL.md frontmatter — only the body.

## How to Run

```bash
docker build -f Dockerfile.base -t adk-autoagent-base .
rm -rf jobs; mkdir -p jobs && uv run harbor run -p tasks/ -n 100 \
  --agent-import-path agent:AutoAgent -o jobs --job-name latest > run.log 2>&1
```

## Logging Results

Log every experiment to `results.tsv`:

```
commit  avg_score  passed  task_scores  cost_usd  status  description
```

`status`: `keep`, `discard`, or `crash`.

## Experiment Loop

1. Check current branch and commit.
2. Read latest `run.log` and task-level results.
3. Diagnose failures — group by root cause.
4. Choose one improvement:
   - `agent.py` editable section (prompt, tools, sub-agents)
   - `skills/<name>/SKILL.md` body
5. Edit, commit, rebuild, rerun.
6. Record in `results.tsv`. Keep or discard.

## Keep / Discard Rules

- `passed` improved → keep.
- `passed` unchanged, harness simpler → keep.
- Otherwise → discard.

## Deepagent Improvement Axes

The agent already has filesystem tools (ls, read_file, write_file, edit_file,
glob, grep) and execute. Add capabilities incrementally as tasks require:

- **Browser tasks**: enable `browser=BrowserConfig(headless=True)` in `create_agent()`
- **HTTP/scraping tasks**: enable `http_tools=True`
- **Complex multi-step tasks**: add `subagents=[SubAgentSpec(...)]` for delegation
- **Skills**: add SKILL.md files to `skills/` for domain-specific guidance

Prefer enabling an existing deepagent capability over writing custom tools.

## NEVER STOP

Once the experiment loop begins, do NOT stop to ask whether to continue.
Keep iterating until the human explicitly interrupts.
