# CLI Guide (`adk-deepagents`)

`adk-deepagents` is the ADK-native CLI for this project. It supports:

- Interactive multi-turn REPL workflows
- Single-turn non-interactive workflows for scripts/CI
- Persistent profile defaults, thread storage, and thread resume
- Human-in-the-loop (HITL) approval for risky tool calls
- Deterministic memory/skills discovery from global + project paths
- Dynamic delegation tools (`task`, `register_subagent`) for runtime worker orchestration
- Workspace filesystem filtering that respects `.gitignore` and skips common dependency caches

> The CLI uses your configured model provider credentials (for example `GOOGLE_API_KEY`)
> when it executes model turns.

## Install

```bash
uv pip install adk-deepagents
# or
uv add adk-deepagents
```

Verify install:

```bash
adk-deepagents --help
python -m adk_deepagents.cli --help
```

## Quick Start

### Interactive mode (default)

```bash
adk-deepagents
```

Start interactive mode and auto-submit the first turn:

```bash
adk-deepagents -m "Summarize this repository"
```

### Non-interactive mode

Run one prompt and exit:

```bash
adk-deepagents -n "Run unit tests and summarize failures"
```

Run with text-only output (automation-friendly):

```bash
adk-deepagents -n "Summarize latest commits" -q
```

Use piped stdin as prompt input:

```bash
printf 'Summarize README.md\n' | adk-deepagents -q
```

Combine piped stdin + `-n` prompt (stdin is prepended):

```bash
printf 'Context from previous step\n' | adk-deepagents -n "Generate release notes" -q
```

## Commands and Flags

### Top-level commands

- `list` — list available local profiles
- `reset --agent <name>` — reset profile memory template (`AGENTS.md`)
- `threads list` / `threads ls` — list persisted threads for a profile
- `threads delete <thread_id>` — delete a persisted thread

### Core flags

| Flag | Description |
|---|---|
| `-a, --agent <name>` | Profile name (default from `~/.adk-deepagents/config.toml`) |
| `--model <name>` | Model override (persisted as new default) |
| `-n, --non-interactive <prompt>` | Run one non-interactive turn |
| `-m, --message <prompt>` | Start interactive mode with an initial prompt |
| `-r, --resume [thread_id]` | Resume latest thread or a specific thread id |
| `-q, --quiet` | Suppress status lines for non-interactive mode |
| `--no-stream` | Buffer non-interactive output and print at end |
| `--auto-approve` | Auto-approve confirmation requests |
| `--shell-allow-list <csv\|recommended>` | Allow specific shell commands in non-interactive mode |
| `-v, --version` | Print CLI version |

### Interactive slash commands

- `/help` — show interactive command help
- `/threads` — list recent threads and show current thread
- `/threads <index|thread_id|latest>` — switch active thread
- `/clear` — create a new thread and make it active
- `/model` — show active model
- `/model <name|default>` — switch model for subsequent turns
- `/quit` or `/q` — exit interactive mode

## Non-interactive Safety Rules

Non-interactive mode is intentionally conservative:

1. **Shell is blocked by default.**
   - Enable explicitly with `--shell-allow-list`.
   - Use `recommended` to expand to a built-in safe-ish command set (`git`, `ls`, `uv`,
     `pytest`, etc.).
2. **Shell control operators are rejected.**
   - Chaining/pipes/redirection tokens (`&&`, `||`, `|`, `>`, `<`, etc.) are blocked.
3. **Confirmation-required tools are blocked unless `--auto-approve` is set.**
   - Applies to `write_file`, `edit_file`, and `delete_file`.

Exit codes:

- `0` = success
- `1` = runtime/config/execution failure
- `2` = usage/validation error

## Deterministic Precedence Rules

- **Model precedence:** `--model` > `ADK_DEEPAGENTS_MODEL` > config default
- **Memory precedence:**
  1. `~/.adk-deepagents/profiles/<agent>/AGENTS.md`
  2. `<cwd>/AGENTS.md`
- **Skills precedence:**
  1. `~/.adk-deepagents/profiles/<agent>/skills`
  2. `<cwd>/skills`

## Manual Smoke Checklist (Release)

Run this checklist on **Linux** and **macOS** before release.

- [ ] `adk-deepagents --help` and `python -m adk_deepagents.cli --help` both work.
- [ ] Interactive mode starts: `adk-deepagents`.
- [ ] Interactive slash commands work: `/help`, `/threads`, `/clear`, `/model`, `/q`.
- [ ] HITL prompt appears for a risky tool call and supports approve/reject decisions.
- [ ] Non-interactive one-shot works: `adk-deepagents -n "hello" -q`.
- [ ] Piped stdin works: `printf 'hello\n' | adk-deepagents -q`.
- [ ] Piped stdin + flag prompt merge works:
      `printf 'context\n' | adk-deepagents -n "task" -q`.
- [ ] Thread resume works: run once, then `adk-deepagents --resume -n "follow-up" -q`.
- [ ] Non-interactive shell is blocked without allow-list.
- [ ] Non-interactive allow-list works with explicit commands and blocks unlisted commands.
- [ ] Required project checks pass:
      `uv run ruff format --check . && uv run ruff check . && uv run ty check && uv run pytest -m "not llm"`.
