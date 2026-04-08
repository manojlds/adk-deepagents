# Data Extraction — Meta-Agent Optimization Example

Demonstrates two modes of agent self-improvement on data extraction tasks.

## Three Modes

### Mode 1: Automated Meta-Loop with Coding Agent (`meta_loop.py`)

A Python script that runs the full outer loop (benchmark → score → keep/discard)
and shells out to a coding agent CLI for the "diagnose and edit" step.
**This is the recommended mode** — it combines Python's reliable scoring
with a coding agent's full editing capabilities.

```bash
# With OpenCode (default)
uv run python -m examples.data_extraction.meta_loop

# With Claude Code
META_AGENT=claude uv run python -m examples.data_extraction.meta_loop

# With Amp
META_AGENT=amp uv run python -m examples.data_extraction.meta_loop

# With a custom command
META_AGENT_CMD='my-agent --prompt "{prompt}"' uv run python -m examples.data_extraction.meta_loop

# Control iterations
MAX_ITERATIONS=10 uv run python -m examples.data_extraction.meta_loop
```

The script:
1. Runs the baseline benchmark
2. Invokes the coding agent with failure diagnostics → it edits `agent.py`
3. Validates the edit (import check)
4. Re-runs the benchmark
5. Keeps or discards (reverts `agent.py`) based on score
6. Records history + learnings
7. Loops until convergence or max iterations
8. Final test-split evaluation

### Mode 2: Manual Coding Agent Loop (`PROGRAM.md`)

Tell your coding agent to read `PROGRAM.md` and it will run the loop itself
(reading task files, running benchmarks via CLI, editing `agent.py`).

```bash
# In your coding agent:
#   "Read examples/data_extraction/PROGRAM.md and start the experiment loop"
```

### Mode 3: Programmatic Optimization Loop (`run.py`)

A Python script that uses the library's `run_optimization_loop()` with an
LLM reflector to automatically suggest and apply prompt improvements.
No coding agent needed.

```bash
GOOGLE_API_KEY=... uv run python -m examples.data_extraction.run
```

## Tasks

| Task | Split | Description |
|---|---|---|
| `invoice_simple` | train | Extract vendor, line items, totals from a text invoice |
| `receipt_multiline` | train | Parse a retail receipt with items and tax |
| `product_listing` | train | Extract product details from an e-commerce listing |
| `server_logs` | train | Summarize access logs (counts, top paths, errors) |
| `email_contacts` | train | Extract contacts from an email thread |
| `csv_malformed` | **test** | Parse CSV with mixed delimiters and quoting |
| `mixed_currency` | **test** | Extract transactions with £/€/¥/$ amounts |
| `nested_tables` | **test** | Extract hierarchical org chart data |

## Verification

Tasks use **deterministic JSON field matching** (not an LLM judge) for scoring:
- Exact field match: full credit
- Present but wrong value: 30% credit
- Missing field: 0 credit
- Numeric tolerance: 0.01
- String comparison: case-insensitive, whitespace-normalized

## CLI Reference

```bash
# Run train benchmark (default)
uv run python examples/data_extraction/agent.py benchmark

# Run test benchmark (held-out — don't read traces!)
uv run python examples/data_extraction/agent.py benchmark --split test

# Run specific tasks
uv run python examples/data_extraction/agent.py benchmark --tasks invoice_simple,receipt_multiline

# Run 3-step acceptance gate
uv run python examples/data_extraction/agent.py gate

# Record a result
uv run python examples/data_extraction/agent.py record --score 0.82 --pass-rate 0.8 --desc "added date format hint"

# Show score history with sparkline
uv run python examples/data_extraction/agent.py history
```

## File Layout

```
examples/data_extraction/
├── PROGRAM.md        ← Natural language program for coding agents (Mode 2)
├── meta_loop.py      ← Automated outer loop with coding agent (Mode 1)
├── agent.py          ← Editable harness (prompt, tools, config) + CLI
├── run.py            ← Programmatic optimization loop (Mode 3)
├── benchmark.py      ← Benchmark runner (loads tasks, runs agent, scores)
├── verify.py         ← Deterministic JSON field matcher
├── tasks/
│   ├── splits.yaml   ← Train/test split definition
│   ├── invoice_simple/
│   │   ├── task.yaml, input.txt, expected.json
│   ├── receipt_multiline/
│   │   ├── ...
│   └── (8 tasks total)
└── workspace/        ← Generated at runtime (gitignored)
    ├── history.jsonl  ← Score progression
    ├── learnings.jsonl ← What worked/failed
    ├── suite.json     ← Self-growing regression suite
    ├── agent.py.original ← Backup of original harness
    └── trajectories/  ← Per-run execution traces
```

## Using with Different Coding Agents

### Amp

```
Read examples/data_extraction/PROGRAM.md and start the experiment loop
```

### OpenCode

```
@read examples/data_extraction/PROGRAM.md
Follow the program and start the experiment loop
```

### Claude Code

```bash
claude "Read examples/data_extraction/PROGRAM.md and start the experiment loop"
```

### Cursor / Windsurf / any coding AI

Open `PROGRAM.md` and `agent.py` in the editor, then ask the AI to follow the program.
