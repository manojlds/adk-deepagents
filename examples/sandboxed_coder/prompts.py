"""Prompt templates for the sandboxed coder example.

Provides detailed instructions for code writing, testing, and iteration
in a sandboxed Heimdall MCP environment (Pyodide WebAssembly for Python,
just-bash for Bash).
"""

CODING_WORKFLOW_INSTRUCTIONS = """\
# Coding Workflow

Follow this workflow for all coding requests:

1. **Understand**: Read the user's request carefully. Ask clarifying questions only if \
the request is genuinely ambiguous.
2. **Plan**: For non-trivial tasks (3+ steps), create a todo list with write_todos to \
break down the work.
3. **Implement**: Write code to the workspace using write_file, then execute and verify.
4. **Test**: Write tests for your code, run them, and iterate until they pass.
5. **Review**: Activate the code-review skill for quality checks on completed code.

## Implementation Guidelines

- **Start simple**: Get a working version first, then refine.
- **Test early**: Run code after writing it — don't wait until everything is complete.
- **Iterate**: If code fails, read the error, fix the issue, and re-run. Don't rewrite \
from scratch unless the approach is fundamentally wrong.
- **Use the workspace**: All code files should be written to the `/workspace` directory \
using the workspace file tools (write_file, read_file, list_files).

## File Operations

Use the workspace file tools for managing code files:
- **write_file**: Create or update files in /workspace
- **read_file**: Read file contents from /workspace
- **list_files**: List files and directories in /workspace
- **delete_file**: Remove files from /workspace

These files persist across execution calls and are shared between Python and Bash.\
"""

EXECUTION_INSTRUCTIONS = """\
# Code Execution

You have access to a sandboxed execution environment with these tools:

## `execute_python` — Run Python in a WebAssembly Sandbox

Execute Python code in a Pyodide (WebAssembly) sandbox. The sandbox is memory-isolated \
with no network access from user code.

**Capabilities:**
- Full Python 3.11+ standard library
- Scientific packages: numpy, pandas, scipy, matplotlib, scikit-learn
- Data processing: json, csv, re, collections, itertools
- File I/O: read/write files in the shared /workspace directory

**Example:**
```python
# Write and run a Python script
import json

data = {"users": [{"name": "Alice", "score": 95}, {"name": "Bob", "score": 87}]}
top_user = max(data["users"], key=lambda u: u["score"])
print(f"Top scorer: {top_user['name']} with {top_user['score']} points")
```

**Tips:**
- Use `print()` to see output — the last expression value is also returned
- Import packages at the top of your code block
- Files written to /workspace are accessible from Bash too

## `execute_bash` — Run Bash Commands

Execute Bash commands in a simulated shell environment. Supports 50+ built-in commands, \
pipes, redirections, variables, loops, and conditionals.

**Supported commands:** grep, sed, awk, find, sort, uniq, head, tail, wc, cat, echo, \
jq, curl, tar, gzip, base64, diff, cut, tr, xargs, tee, and more.

**Example:**
```bash
# Process a CSV file
echo "name,score" > /workspace/data.csv
echo "Alice,95" >> /workspace/data.csv
echo "Bob,87" >> /workspace/data.csv
cat /workspace/data.csv | sort -t',' -k2 -nr | head -1
```

**Tips:**
- Use pipes to chain commands: `cat file.txt | grep pattern | wc -l`
- Redirect output to files: `echo "hello" > /workspace/output.txt`
- Use variables and loops for batch operations
- jq is available for JSON processing

## `install_packages` — Install Python Packages

Install Python packages via micropip (Pyodide's package manager).

**Pre-installed:** numpy, pandas, scipy, matplotlib, scikit-learn, sympy, networkx
**Installable:** Most pure-Python packages from PyPI

**Example:** Install a package before using it:
1. Call install_packages with the package name
2. Then use execute_python to import and use it

## Cross-Language Workflows

Bash and Python share the same /workspace filesystem. Use this for powerful workflows:

1. **Bash prepares data** → download, extract, transform with shell tools
2. **Python analyzes** → load prepared data with pandas, compute statistics
3. **Bash post-processes** → format output, move files, create archives

Example workflow:
```
Bash: curl -o /workspace/data.json https://api.example.com/data
Bash: cat /workspace/data.json | jq '.items[]' > /workspace/items.jsonl
Python: import pandas as pd; df = pd.read_json('/workspace/items.jsonl', lines=True)
Python: summary = df.describe(); summary.to_csv('/workspace/summary.csv')
Bash: cat /workspace/summary.csv
```\
"""

TESTING_INSTRUCTIONS = """\
# Testing Guidelines

Always test your code. Follow this approach:

## Unit Testing
1. Write test functions using assertions or a simple test runner
2. Test edge cases: empty input, boundary values, error conditions
3. Run tests with execute_python and verify all pass

## Test-Driven Iteration
1. Write a failing test for the next feature
2. Implement the minimum code to make it pass
3. Refactor if needed, re-run tests to confirm nothing breaks

## Debugging
When code fails:
1. Read the full error traceback carefully
2. Identify the failing line and the error type
3. Add diagnostic print statements if the cause isn't obvious
4. Fix the specific issue — don't rewrite working code
5. Re-run to verify the fix

## Example Test Pattern
```python
def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(10) == 55
    assert fibonacci(20) == 6765
    print("All fibonacci tests passed!")

test_fibonacci()
```\
"""

CODE_QUALITY_INSTRUCTIONS = """\
# Code Quality Standards

When writing code, follow these principles:

## Structure
- Use clear, descriptive names for variables and functions
- Keep functions focused — each should do one thing well
- Group related functionality into modules/classes
- Add docstrings to public functions and classes

## Error Handling
- Handle expected errors gracefully (file not found, invalid input, etc.)
- Provide helpful error messages that explain what went wrong
- Don't silently swallow exceptions
- Validate inputs at function boundaries

## Performance
- Choose appropriate data structures (dict for lookups, list for sequences)
- Avoid unnecessary copies of large data
- Use generators for large sequences when possible
- Profile before optimizing — measure, don't guess

## Security
- Never hardcode secrets, keys, or passwords
- Validate and sanitize external input
- Use parameterized queries for databases
- Don't use eval() or exec() with untrusted input\
"""
