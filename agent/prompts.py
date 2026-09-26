"""Prompt templates and system guidelines for the Quantitative Finance Agent.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are an expert Quantitative Finance Engineer and Python Developer participating in Agenthon 2026 / QFBench 2.0 Track 1.

Your objective is to write robust, bug-free, self-contained Python code that solves the quantitative finance task described in the prompt and outputs the required deliverable files.

CRITICAL OPERATING RULES:
1. Deliverable Output Path:
   - All deliverables (e.g. results.parquet, output.csv, predictions.json) MUST be written to the specified output directory.
   - Both `/app/output` and the designated `--out` directory are mounted to the same destination. Ensure output files are saved there.
   - Do NOT write reward.json or pytest_report.json (the test harness generates these offline).

2. Input Files:
   - Input datasets are provided under the task directory, typically in `data/` or `environment/data/`.
   - Your code should dynamically locate and read input files from the provided task path.

3. Numerical Precision & Domain Invariants:
   - Ensure calculations respect financial invariants (e.g. put-call parity, non-negative volatilities/variances, correct Greek signs, valid date alignments).
   - Guard against division by zero, NaN, Inf, and empty datasets.
   - When writing Parquet/CSV files, ensure column names and types match the instruction specification EXACTLY.

4. Libraries:
   - You have access to: numpy, pandas, scipy, statsmodels, scikit-learn, pyarrow, arch, polars, ta-lib.
   - Network access is disabled. Do not try to download external packages or fetch online data.

5. Code Formatting:
   - Provide the complete, runnable Python script inside a single ```python ... ``` markdown block.
   - Do NOT output partial snippets or placeholders. The code will be executed directly via Python.
"""


def build_initial_prompt(
    instruction_text: str,
    task_dir: str,
    out_dir: str,
    discovered_files: list[str],
) -> str:
    """Format the initial prompt for the coding agent."""
    files_str = "\n".join(f"- {f}" for f in discovered_files) if discovered_files else "None discovered"
    return f"""Please solve the following quantitative finance task.

### TASK SPECIFICATION:
{instruction_text}

### RUNTIME ENVIRONMENT:
- Task Directory (Read-only inputs): `{task_dir}`
- Output Directory (Deliverables must be written here): `{out_dir}`
- Discovered Input Files:
{files_str}

### INSTRUCTIONS:
1. Write a complete, standalone Python script to perform the calculations and save the required deliverables to `{out_dir}`.
2. Ensure all column names, file formats, and data structures match the specification exactly.
3. Enclose your complete script inside a ```python ... ``` code block.
"""


def build_repair_prompt(
    previous_code: str,
    error_message: str,
    out_dir: str,
    missing_deliverables: list[str] | None = None,
) -> str:
    """Format a repair prompt when execution fails or deliverables are missing."""
    missing_str = ""
    if missing_deliverables:
        missing_str = f"\nMissing Expected Deliverables in `{out_dir}`:\n" + "\n".join(
            f"- {f}" for f in missing_deliverables
        )

    return f"""The previous Python script failed to generate the required deliverables.

### PREVIOUS CODE:
```python
{previous_code}
```

### EXECUTION ERROR / OUTPUT:
```
{error_message}
```
{missing_str}

### INSTRUCTIONS:
1. Carefully diagnose the error (e.g. schema mismatch, file path issue, NaN/inf computation, or missing column).
2. Fix the bug and provide the entire corrected Python script.
3. Make sure all deliverables are successfully written to `{out_dir}`.
4. Enclose your corrected script inside a single ```python ... ``` code block.
"""
