# Meta-Agent Optimization: Detailed Plan

> Bringing self-improving agent harness patterns from meta-agent, autoagent, and auto-harness
> into adk-deepagents — both as **library primitives** and a concrete **data extraction
> benchmark example**.

---

## Table of Contents

1. [Current State](#1-current-state)
2. [What Goes Into the Library](#2-what-goes-into-the-library)
3. [End-to-End Example: Data Extraction Benchmark](#3-end-to-end-example-data-extraction-benchmark)
4. [Implementation Phases](#4-implementation-phases)
5. [File Layout](#5-file-layout)

---

## 1. Current State

### What we already have (`adk_deepagents/optimization/`)

| Module | What it does | Status |
|---|---|---|
| `trajectory.py` | `Trajectory`, `AgentStep`, `ModelCall`, `ToolCall`, `FeedbackEntry` dataclasses | ✅ Complete |
| `store.py` | `TrajectoryStore` — JSON-backed persistence with index, filtering, golden marking, JSONL export | ✅ Complete |
| `evaluator.py` | LLM judge with configurable `EvaluationRubric`, weighted scoring, majority voting | ✅ Complete |
| `replay.py` | Re-execute trajectories with tool approval, user simulator, ephemeral instructions | ✅ Complete |
| `loop.py` | `run_optimization_loop()` — replay→evaluate→reflect→apply suggestions | ✅ Complete |

### What we already have elsewhere

| Module | Relevance |
|---|---|
| `telemetry/trace_reader.py` | Reads OTEL JSON traces → `Trajectory` objects |
| `harbor/` | Harbor benchmark adapter (`HarborAdapter`, `HarborBackend`, execute tool) |
| `examples/optimization_loop/` | Content-writing optimization demo (prompt-only) |
| `examples/autoagent/` | Harbor-based autoagent harness + `program.md` |

### Gaps identified from meta-agent / autoagent / auto-harness

| Gap | Inspired by | Priority |
|---|---|---|
| No gating — changes are always applied | auto-harness 3-step gate | High |
| No regression suite — no protection against regressions | auto-harness `suite.json` | High |
| No train/test split — evaluator sees all data | auto-harness, meta-agent holdout | High |
| No benchmark runner abstraction — loop works on trajectories only | auto-harness `BenchmarkRunner`, meta-agent `eval_runner.py` | High |
| No experience query tools — store not exposed as agent tools | meta-agent CLI (`list`, `failures`, `diff`, `pareto`) | Medium |
| No persistent learnings — cross-session memory lost | auto-harness `learnings.md` | Medium |
| No harness-as-code — suggestions are `agent_kwargs` dicts, not runnable files | autoagent `agent.py`, meta-agent `config.py` | Medium |
| No monotonic score tracking — `results.tsv` equivalent | auto-harness `record.py` | Medium |
| No two-level skill evolution — reflector prompt is static | meta-agent skill evolver | Low (Phase 3) |
| No simplicity bias — no penalty for complexity | autoagent keep/discard rules | Low |

---

## 2. What Goes Into the Library

### 2.1 Benchmark Runner Abstraction

**Module:** `adk_deepagents/optimization/benchmark.py`

An abstract base class that decouples the optimization loop from any specific
benchmark. Inspired by auto-harness's `BenchmarkRunner` ABC.

```python
@dataclass
class TaskResult:
    """Result of running a single benchmark task."""
    task_id: str
    reward: float                    # 0.0–1.0
    passed: bool                     # reward > threshold
    trajectory: Trajectory | None    # captured execution trace
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    verify_output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated result of running a full benchmark."""
    task_results: dict[str, TaskResult]   # task_id → result
    pass_rate: float                       # fraction passed
    mean_reward: float
    total_cost_usd: float = 0.0
    split: str = "train"                   # "train" or "test"


class BenchmarkRunner(ABC):
    """Abstract benchmark runner."""

    @abstractmethod
    async def run(
        self,
        agent_factory: Callable[[], LlmAgent],
        *,
        task_ids: list[str] | None = None,
        split: str = "train",
    ) -> BenchmarkResult: ...

    @abstractmethod
    def list_task_ids(self, *, split: str = "train") -> list[str]: ...
```

**Concrete implementations provided:**

- `LocalBenchmarkRunner` — runs tasks defined as YAML/JSON files with
  `instruction`, optional `workspace` directory, and a `verify` command
  (exit code 0 = pass). This is the simplest runner for custom benchmarks.

- `TrajectoryBenchmarkRunner` — wraps the existing replay-based approach:
  given seed trajectories, replays them through the agent and evaluates
  with the LLM judge. This bridges the existing `run_optimization_loop()`
  approach into the new benchmark abstraction.

### 2.2 Gating System

**Module:** `adk_deepagents/optimization/gate.py`

A three-step gate inspired by auto-harness, adapted for the ADK optimization
loop. Every candidate must pass all steps before being accepted.

```python
@dataclass
class GateConfig:
    """Configuration for the acceptance gate."""
    regression_threshold: float = 0.8     # suite pass rate minimum
    require_improvement: bool = True      # must beat or match historical best
    auto_promote: bool = True             # auto-add newly passing tasks to suite

@dataclass
class RegressionSuite:
    """Self-growing regression suite backed by JSON."""
    path: Path
    tasks: dict[str, RegressionTask]      # task_id → last known result

    def load(cls, path: Path) -> RegressionSuite: ...
    def save(self) -> None: ...
    def promote(self, task_ids: list[str]) -> int: ...

@dataclass
class GateResult:
    passed: bool
    step_results: list[GateStepResult]
    promoted_tasks: list[str]

async def run_gate(
    candidate: OptimizationCandidate,
    agent_factory: Callable[[], LlmAgent],
    *,
    train_runner: BenchmarkRunner,
    test_runner: BenchmarkRunner,
    suite: RegressionSuite,
    history: ScoreHistory,
    config: GateConfig = GateConfig(),
) -> GateResult:
    """Three-step gate:
    1. Regression suite (train split) — pass_rate >= threshold
    2. Full benchmark (test split) — val_score >= historical best
    3. Suite promotion — re-run previously-failing train tasks, promote passes
    """
```

### 2.3 Score History & Audit Trail

**Module:** `adk_deepagents/optimization/history.py`

Monotonic score tracking inspired by auto-harness's `results.tsv` and
meta-agent's `history.json`.

```python
@dataclass
class HistoryEntry:
    iteration: int
    val_score: float
    pass_rate: float
    cost_usd: float
    candidate_hash: str           # hash of agent_kwargs for dedup
    accepted: bool
    description: str
    timestamp: str
    train_scores: dict[str, float] | None = None
    test_scores: dict[str, float] | None = None

class ScoreHistory:
    """Append-only score history backed by JSONL."""
    def __init__(self, path: Path): ...
    def append(self, entry: HistoryEntry) -> None: ...
    def best_val_score(self) -> float | None: ...
    def entries(self) -> list[HistoryEntry]: ...
    def sparkline(self) -> str: ...
```

### 2.4 Learnings Store (Cross-Session Memory)

**Module:** `adk_deepagents/optimization/learnings.py`

Persistent cross-session memory inspired by auto-harness's `learnings.md`.
Structured rather than freeform, so the reflector can consume it
programmatically.

```python
@dataclass
class LearningEntry:
    iteration: int
    category: Literal[
        "confirmed_pattern",    # a failure pattern confirmed across 2+ tasks
        "successful_change",    # a change that improved scores
        "failed_attempt",       # a change that was gated/reverted
        "open_question",        # something needing human input
    ]
    summary: str
    evidence_trace_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

class LearningsStore:
    """JSONL-backed learnings store."""
    def __init__(self, path: Path): ...
    def append(self, entry: LearningEntry) -> None: ...
    def recent(self, n: int = 20) -> list[LearningEntry]: ...
    def by_category(self, category: str) -> list[LearningEntry]: ...
    def to_prompt_context(self, max_entries: int = 10) -> str: ...
```

The reflector (in `loop.py`) is updated to:
1. Read learnings before suggesting changes (avoid repeating failed attempts)
2. Write learnings after each iteration (record what worked/failed)
3. Inject learnings summary into the reflector prompt

### 2.5 Experience Query Tools

**Module:** `adk_deepagents/optimization/tools.py`

Expose `TrajectoryStore` and `ScoreHistory` as ADK tools so a meta-agent
can introspect its own history. Inspired by meta-agent's CLI.

```python
def create_experience_tools(
    store: TrajectoryStore,
    history: ScoreHistory,
    learnings: LearningsStore | None = None,
) -> list[Callable]:
    """Create tools for querying the experience store."""

    def list_candidates(*, sort_by: str = "score") -> str:
        """List all evaluated candidates ranked by score."""
        ...

    def show_failures(*, trace_id: str) -> str:
        """Show failed tool calls and issues for a trajectory."""
        ...

    def diff_candidates(*, trace_id_a: str, trace_id_b: str) -> str:
        """Show which tasks flipped pass/fail between two candidates."""
        ...

    def show_learnings(*, category: str | None = None, n: int = 10) -> str:
        """Show recent learnings from the optimization history."""
        ...

    def score_history(*, last_n: int = 20) -> str:
        """Show score progression across iterations."""
        ...

    return [list_candidates, show_failures, diff_candidates, show_learnings, score_history]
```

### 2.6 Enhanced Optimization Loop

**Module:** Updates to `adk_deepagents/optimization/loop.py`

Extend `run_optimization_loop()` with the new primitives:

```python
async def run_optimization_loop(
    # --- existing params ---
    trajectories: Sequence[Trajectory],
    base_candidate: OptimizationCandidate,
    agent_builder_factory: ...,
    evaluator_model: str = "gemini-2.5-flash",
    ...
    # --- new params ---
    benchmark: BenchmarkRunner | None = None,        # replaces trajectory-only mode
    test_benchmark: BenchmarkRunner | None = None,    # held-out eval (train/test split)
    gate_config: GateConfig | None = None,            # gating system
    suite_path: Path | None = None,                   # regression suite file
    history_path: Path | None = None,                 # score history file
    learnings_path: Path | None = None,               # learnings store file
    one_change_per_iteration: bool = True,             # auto-harness discipline
) -> OptimizationResult:
```

When `benchmark` is provided, the loop switches from trajectory-replay mode
to benchmark-runner mode:

1. **Run train benchmark** → get `BenchmarkResult` with per-task trajectories
2. **Run gate** (if `gate_config` is set) → regression suite + test benchmark
3. **Evaluate** — use LLM judge on failed trajectories (or all trajectories)
4. **Reflect** — generate suggestions, injecting learnings context
5. **Apply** — one suggestion at a time if `one_change_per_iteration=True`
6. **Record** — append to history, write learnings
7. **Accept/revert** — based on gate result

Backward compatible: when `benchmark` is `None`, the existing trajectory-based
flow is used unchanged.

---

## 3. End-to-End Example: Data Extraction Benchmark

**Directory:** `examples/data_extraction/`

A complete, runnable example that demonstrates the full meta-agent optimization
loop on a **data extraction** task: given messy semi-structured text files
(invoices, receipts, product listings, log files), extract structured JSON.

### 3.1 Why Data Extraction

- **Deterministic verification** — expected output is a known JSON; we can
  diff for exact field matching (no LLM judge needed for scoring)
- **Measurable improvement** — field-level precision/recall gives a continuous
  0.0–1.0 reward signal
- **Rich optimization surface** — the agent needs prompt engineering (what
  fields to extract, output format), tool usage (read_file, grep), and
  potentially multi-step reasoning
- **Real-world relevance** — data extraction is a common agent task

### 3.2 Task Design

Each task is a directory containing:

```
tasks/
├── invoice_simple/
│   ├── input.txt          # messy invoice text
│   ├── expected.json      # ground-truth structured output
│   └── task.yaml          # instruction + verify config
├── receipt_multiline/
│   ├── input.txt
│   ├── expected.json
│   └── task.yaml
├── product_listing/
│   ├── input.txt
│   ├── expected.json
│   └── task.yaml
├── server_logs/
│   ├── input.txt
│   ├── expected.json
│   └── task.yaml
├── email_contacts/
│   ├── input.txt
│   ├── expected.json
│   └── task.yaml
├── csv_malformed/
│   ├── input.txt
│   ├── expected.json
│   └── task.yaml
├── mixed_currency/
│   ├── input.txt
│   ├── expected.json
│   └── task.yaml
└── nested_tables/
    ├── input.txt
    ├── expected.json
    └── task.yaml
```

**Train/test split:** Tasks 1-5 are train, tasks 6-8 are test (held out).

**task.yaml format:**

```yaml
name: invoice_simple
instruction: |
  Read the file /workspace/input.txt and extract all invoice data
  as structured JSON. Write the result to /workspace/output.json.
  The JSON should contain: vendor, invoice_number, date, line_items
  (each with description, quantity, unit_price, total), subtotal,
  tax, and grand_total.
input_file: input.txt
expected_file: expected.json
output_file: output.json
```

### 3.3 Verifier

**File:** `examples/data_extraction/verify.py`

A deterministic JSON field-matching verifier that produces a 0.0–1.0 reward:

```python
def verify(expected_path: str, actual_path: str) -> float:
    """Compare expected and actual JSON, return field-level match score.

    Scoring:
    - Each top-level field present and correct: +1
    - Each top-level field present but wrong value: +0.3
    - Each top-level field missing: +0
    - For array fields (line_items): compare element-by-element
    - Numeric fields: allow 0.01 tolerance
    - String fields: case-insensitive, whitespace-normalized

    Returns: score between 0.0 and 1.0
    """
```

### 3.4 Benchmark Runner

**File:** `examples/data_extraction/benchmark.py`

A `LocalBenchmarkRunner` subclass that:

1. For each task: copies `input.txt` to a temp workspace
2. Runs the agent with the task instruction
3. Reads `/workspace/output.json` from the agent's backend
4. Runs `verify.py` to score the output against `expected.json`
5. Captures the full trajectory

```python
class DataExtractionBenchmark(BenchmarkRunner):
    def __init__(self, tasks_dir: Path, split: str = "train"):
        self.tasks = load_tasks(tasks_dir, split=split)

    async def run(self, agent_factory, *, task_ids=None, split="train"):
        results = {}
        for task in self.tasks:
            # 1. Create agent with FilesystemBackend pointed at temp workspace
            # 2. Copy input.txt to workspace
            # 3. Run agent with task instruction
            # 4. Read output.json from workspace
            # 5. Score with verify()
            # 6. Capture trajectory
            results[task.id] = TaskResult(...)
        return BenchmarkResult(task_results=results, ...)
```

### 3.5 The Optimization Script

**File:** `examples/data_extraction/run.py`

```python
"""Data extraction meta-agent optimization.

Demonstrates:
1. A benchmark with deterministic verification (no LLM judge for scoring)
2. Train/test split with gating
3. Self-growing regression suite
4. Score history tracking
5. Learnings persistence
6. Experience query tools for the reflector

Usage:
    GOOGLE_API_KEY=... uv run python examples/data_extraction/run.py
"""

async def main():
    model = resolve_model()
    workspace = Path("examples/data_extraction/workspace")
    workspace.mkdir(exist_ok=True)

    # 1. Set up benchmarks (train/test split)
    train_benchmark = DataExtractionBenchmark(
        tasks_dir=Path("examples/data_extraction/tasks"),
        split="train",
    )
    test_benchmark = DataExtractionBenchmark(
        tasks_dir=Path("examples/data_extraction/tasks"),
        split="test",
    )

    # 2. Set up optimization infrastructure
    store = TrajectoryStore(workspace / "trajectories")
    history = ScoreHistory(workspace / "history.jsonl")
    learnings = LearningsStore(workspace / "learnings.jsonl")
    suite = RegressionSuite.load(workspace / "suite.json")

    # 3. Initial agent config (intentionally minimal)
    base_candidate = OptimizationCandidate(
        agent_kwargs={
            "name": "data_extractor",
            "instruction": (
                "You are a data extraction assistant. "
                "Read input files and extract structured JSON data."
            ),
            "execution": "local",
        }
    )

    # 4. Agent builder
    def agent_builder_factory(candidate):
        agent = create_deep_agent(**{**candidate.agent_kwargs, "model": model})
        return BuiltAgent(agent=agent)

    # 5. Evaluation rubric (task-specific)
    rubric = EvaluationRubric(
        criteria=[
            EvaluationCriterion(
                name="extraction_completeness",
                description="Did the agent extract ALL fields from the input?",
                weight=0.4,
            ),
            EvaluationCriterion(
                name="format_correctness",
                description="Is the output valid JSON matching the expected schema?",
                weight=0.3,
            ),
            EvaluationCriterion(
                name="value_accuracy",
                description="Are extracted values correct (numbers, dates, names)?",
                weight=0.3,
            ),
        ],
        name="data_extraction_v1",
    )

    # 6. Run optimization
    result = await run_optimization_loop(
        trajectories=[],                    # not using trajectory-only mode
        base_candidate=base_candidate,
        agent_builder_factory=agent_builder_factory,
        evaluator_model=model,
        rubric=rubric,
        benchmark=train_benchmark,
        test_benchmark=test_benchmark,
        gate_config=GateConfig(
            regression_threshold=0.8,
            require_improvement=True,
            auto_promote=True,
        ),
        suite_path=workspace / "suite.json",
        history_path=workspace / "history.jsonl",
        learnings_path=workspace / "learnings.jsonl",
        store=store,
        max_iterations=5,
        apply_mode="prompt_and_skills",
        one_change_per_iteration=True,
    )

    # 7. Report
    print(f"\nBest score: {history.best_val_score()}")
    print(f"Score progression: {history.sparkline()}")
    print(f"Regression suite: {len(suite.tasks)} tasks")
    print(f"Learnings: {len(learnings.recent(100))} entries")
```

### 3.6 Example Data Files

**`tasks/invoice_simple/input.txt`:**

```
INVOICE
========================================
From: Acme Corp, 123 Main St, Springfield
To: Widget Inc, 456 Oak Ave, Shelbyville

Invoice #: INV-2025-0042
Date: January 15, 2025
Due: February 14, 2025

Items:
  1. Widget Assembly Kit (x3) ........... $45.00 each
  2. Premium Bolts Pack (x10) ........... $12.50 each
  3. Shipping & Handling ................ $15.00

                              Subtotal: $275.00
                              Tax (8%):  $22.00
                              TOTAL:    $297.00

Payment terms: Net 30
```

**`tasks/invoice_simple/expected.json`:**

```json
{
  "vendor": "Acme Corp",
  "invoice_number": "INV-2025-0042",
  "date": "2025-01-15",
  "due_date": "2025-02-14",
  "line_items": [
    {"description": "Widget Assembly Kit", "quantity": 3, "unit_price": 45.00, "total": 135.00},
    {"description": "Premium Bolts Pack", "quantity": 10, "unit_price": 12.50, "total": 125.00},
    {"description": "Shipping & Handling", "quantity": 1, "unit_price": 15.00, "total": 15.00}
  ],
  "subtotal": 275.00,
  "tax": 22.00,
  "grand_total": 297.00
}
```

---

## 4. Implementation Phases

### Phase 1: Library Primitives (benchmark, gate, history)

**Files to create:**

| File | Description |
|---|---|
| `adk_deepagents/optimization/benchmark.py` | `BenchmarkRunner` ABC + `TaskResult` + `BenchmarkResult` + `LocalBenchmarkRunner` + `TrajectoryBenchmarkRunner` |
| `adk_deepagents/optimization/gate.py` | `GateConfig` + `RegressionSuite` + `GateResult` + `run_gate()` |
| `adk_deepagents/optimization/history.py` | `ScoreHistory` + `HistoryEntry` |
| `tests/unit_tests/optimization/test_benchmark.py` | Unit tests for benchmark runner |
| `tests/unit_tests/optimization/test_gate.py` | Unit tests for gating |
| `tests/unit_tests/optimization/test_history.py` | Unit tests for score history |

**Files to modify:**

| File | Changes |
|---|---|
| `adk_deepagents/optimization/__init__.py` | Export new types |

### Phase 2: Learnings + Experience Tools + Loop Integration

**Files to create:**

| File | Description |
|---|---|
| `adk_deepagents/optimization/learnings.py` | `LearningsStore` + `LearningEntry` |
| `adk_deepagents/optimization/tools.py` | Experience query tools |
| `tests/unit_tests/optimization/test_learnings.py` | Unit tests |
| `tests/unit_tests/optimization/test_tools.py` | Unit tests |

**Files to modify:**

| File | Changes |
|---|---|
| `adk_deepagents/optimization/loop.py` | Add benchmark-runner mode, gating integration, learnings injection, `one_change_per_iteration` |
| `adk_deepagents/optimization/__init__.py` | Export new types |

### Phase 3: Data Extraction Example

**Files to create:**

| File | Description |
|---|---|
| `examples/data_extraction/run.py` | Main optimization script |
| `examples/data_extraction/benchmark.py` | `DataExtractionBenchmark` runner |
| `examples/data_extraction/verify.py` | Deterministic JSON field verifier |
| `examples/data_extraction/tasks/` | 8 task directories with input/expected/task.yaml |
| `examples/data_extraction/README.md` | Documentation |
| `examples/data_extraction/__init__.py` | Package marker |

### Phase 4: Two-Level Evolution (stretch)

**Files to create:**

| File | Description |
|---|---|
| `adk_deepagents/optimization/meta_reflector.py` | Rewrites reflector prompt based on reflector trace effectiveness |

**Files to modify:**

| File | Changes |
|---|---|
| `adk_deepagents/optimization/loop.py` | Add meta-reflector call every N iterations |

---

## 5. File Layout

```
adk_deepagents/optimization/
├── __init__.py              # (updated — exports new types)
├── benchmark.py             # NEW — BenchmarkRunner ABC + implementations
├── evaluator.py             # (existing — unchanged)
├── gate.py                  # NEW — 3-step gating system
├── history.py               # NEW — ScoreHistory + HistoryEntry
├── learnings.py             # NEW — LearningsStore + LearningEntry
├── loop.py                  # (updated — benchmark mode, gating, learnings)
├── replay.py                # (existing — unchanged)
├── store.py                 # (existing — unchanged)
├── tools.py                 # NEW — experience query tools
└── trajectory.py            # (existing — unchanged)

examples/data_extraction/
├── __init__.py
├── README.md
├── run.py                   # Main optimization script
├── benchmark.py             # DataExtractionBenchmark runner
├── verify.py                # JSON field-matching verifier
├── workspace/               # Created at runtime (gitignored)
└── tasks/
    ├── train.yaml           # Lists train-split task IDs
    ├── test.yaml            # Lists test-split task IDs
    ├── invoice_simple/
    │   ├── input.txt
    │   ├── expected.json
    │   └── task.yaml
    ├── receipt_multiline/
    │   ├── ...
    ├── product_listing/
    │   ├── ...
    ├── server_logs/
    │   ├── ...
    ├── email_contacts/
    │   ├── ...
    ├── csv_malformed/
    │   ├── ...
    ├── mixed_currency/
    │   ├── ...
    └── nested_tables/
        ├── ...

tests/unit_tests/optimization/
├── test_benchmark.py        # NEW
├── test_gate.py             # NEW
├── test_history.py          # NEW
├── test_learnings.py        # NEW
├── test_tools.py            # NEW
├── test_evaluator.py        # (existing)
├── test_loop.py             # (updated — tests for new loop modes)
├── test_replay.py           # (existing)
├── test_store.py            # (existing)
└── test_trajectory.py       # (existing)
```

---

## Summary: What's Library vs. What's Example

| Component | Location | Rationale |
|---|---|---|
| `BenchmarkRunner` ABC | **Library** | Reusable across any benchmark |
| `LocalBenchmarkRunner` | **Library** | General-purpose local task runner |
| `TrajectoryBenchmarkRunner` | **Library** | Bridges existing trajectory-based flow |
| `GateConfig` + `run_gate()` | **Library** | Reusable gating primitive |
| `RegressionSuite` | **Library** | Self-growing suite is benchmark-agnostic |
| `ScoreHistory` | **Library** | Audit trail for any optimization run |
| `LearningsStore` | **Library** | Cross-session memory for any agent |
| Experience query tools | **Library** | Any meta-agent can introspect history |
| Enhanced `run_optimization_loop()` | **Library** | Backward-compatible extension |
| `DataExtractionBenchmark` | **Example** | Task-specific benchmark implementation |
| `verify.py` | **Example** | Task-specific verifier |
| Task data files | **Example** | Concrete test data |
| `run.py` | **Example** | Demonstration of all library primitives together |
