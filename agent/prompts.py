"""Prompt templates and system guidelines for the Quantitative Finance Agent."""

from __future__ import annotations

import sys
from importlib import metadata


def _runtime_versions() -> str:
    """Versions of the interpreter that executes generated scripts (the agent's own)."""
    found = [f"Python {sys.version.split()[0]}"]
    for dist in ("pandas", "numpy", "scipy", "pyarrow", "statsmodels", "scikit-learn"):
        try:
            found.append(f"{dist} {metadata.version(dist)}")
        except metadata.PackageNotFoundError:
            continue
    return ", ".join(found)


SYSTEM_PROMPT = """You are an expert Quantitative Finance Engineer and Python Developer participating in Agenthon 2026 / QFBench 2.0 Track 1.

Your objective is to write robust, bug-free, self-contained Python code that solves the quantitative finance task described in the prompt and outputs the required deliverable files.

CRITICAL OPERATING RULES:
1. Paths (IMPORTANT):
   - Task instructions refer to `/input/...` and `/output/...`. At run time these map to the
     directories in the environment variables TASK_DIR and OUTPUT_DIR. Always resolve paths as
     `pathlib.Path(os.environ["TASK_DIR"]) / "<path relative to /input>"` and
     `pathlib.Path(os.environ["OUTPUT_DIR"]) / "<file name>"`. Never hard-code `/input` or `/output`.
   - The script runs from a scratch directory; do not rely on the current working directory.
   - Write only the requested deliverables to OUTPUT_DIR. Do NOT write reward.json or
     pytest_report.json (the test harness generates these offline).

2. Input Files:
   - Input datasets live under TASK_DIR, typically in `environment/data/` or `data/`.

3. Numerical Precision & Domain Invariants:
   - Ensure calculations respect financial invariants (e.g. put-call parity, non-negative volatilities/variances, correct Greek signs, valid date alignments).
   - Guard against division by zero, NaN, Inf, and empty datasets.
   - When writing Parquet/CSV files, ensure column names and types match the instruction specification EXACTLY.

4. Libraries:
   - You have access to: numpy, pandas, scipy, statsmodels, scikit-learn, pyarrow, arch, polars, ta-lib.
   - Installed versions: {versions}. Write code for these versions (pandas 3 removed
     fillna(method=...): use .ffill()/.bfill(); prefer explicit dtypes and pd.to_datetime).
   - Network access is disabled. Do not try to download external packages or fetch online data.
   - Inspect the data previews in the prompt and use the exact column names and JSON keys shown.

5. Response Budget and Formatting:
   - Your whole response (reasoning included) is capped at about 4,000 tokens. Keep reasoning
     short and spend the budget on the script. Keep the script compact (well under 200 lines),
     with few comments and no long docstrings.
   - Provide the complete, runnable Python script inside a single ```python ... ``` markdown block.
     Put nothing after the closing fence.
   - Do NOT output partial snippets or placeholders. The code will be executed directly via Python.
   - Use only ASCII characters in the code (no unicode math symbols such as the partial sign or sigma).
""".replace("{versions}", _runtime_versions())


_MAX_PATH_MAP_LINES = 40

STRUCTURE_INSTRUCTION = """4. Structure the script as one function per deliverable that computes it and writes it to
   OUTPUT_DIR immediately. Load inputs once and pass them in. In main(), call each deliverable
   function inside try/except, print the full traceback of any failure, continue with the rest,
   and finally exit with status 1 if any deliverable failed (so every error is reported at once).
"""

EDIT_INSTRUCTIONS = """Reply with targeted edits to the script, NOT a full rewrite (it is long, and a
rewrite would overrun your ~4,000-token reply cap). Use one or more blocks of exactly this form:

<<<<<<< SEARCH
lines copied EXACTLY from the current script (enough lines to be unique)
=======
the replacement lines
>>>>>>> REPLACE

Each SEARCH must match the script character for character, including indentation. Only if the
script needs a complete redesign, return the whole new script in one ```python ... ``` block."""


def build_initial_prompt(
    instruction_text: str,
    task_dir: str,
    out_dir: str,
    discovered_files: list[str],
    path_map: list[tuple[str, str]] | None = None,
    input_previews: str = "",
    domain_notes: str = "",
    reference_example: str = "",
    structured: bool = False,
    task_facts: str = "",
) -> str:
    """Format the initial prompt for the coding agent (the task prompt reused by every turn)."""
    notes_str = (
        f"\n### DOMAIN NOTES (grader-checked invariants and common mistakes for this kind of "
        f"task):\n{domain_notes}\n"
        if domain_notes
        else ""
    )
    files_str = (
        "\n".join(f"- {f}" for f in discovered_files) if discovered_files else "None discovered"
    )
    map_str = ""
    if path_map:
        shown = path_map[:_MAX_PATH_MAP_LINES]
        more = len(path_map) - len(shown)
        map_str = (
            "- Paths the task specification cites do NOT exist at run time. Read each one from "
            "its TASK_DIR file instead (note the file may have a different name or extension):\n"
            + "\n".join(f"  - `{cited}` -> TASK_DIR/`{rel}`" for cited, rel in shown)
            + (f"\n  - ... and {more} more under the same directories" if more > 0 else "")
            + "\n"
        )
    return f"""Please solve the following quantitative finance task.

### TASK SPECIFICATION:
{instruction_text}

{f"### TASK CONTEXT:{chr(10)}{task_facts}{chr(10)}{chr(10)}" if task_facts else ""}### RUNTIME ENVIRONMENT:
- `/input` in the task specification = env var TASK_DIR (currently `{task_dir}`, read-only)
- `/output` in the task specification = env var OUTPUT_DIR (currently `{out_dir}`)
{map_str}- Discovered input files (relative to TASK_DIR):
{files_str}

### INPUT DATA PREVIEWS:
{input_previews or "(none)"}
{notes_str}{reference_example}
### INSTRUCTIONS:
1. Write a complete, standalone Python script that reads inputs from TASK_DIR and writes the required deliverables to OUTPUT_DIR.
2. Ensure all column names, file formats, and data structures match the specification exactly.
3. Keep reasoning brief, then enclose your complete script inside one ```python ... ``` code block.
{STRUCTURE_INSTRUCTION if structured else ""}"""


def build_repair_prompt(
    task_prompt: str,
    previous_code: str,
    error_message: str,
    out_dir: str,
    missing_deliverables: list[str] | None = None,
    headline: str = "Your previous Python script for this task failed to generate the required deliverables.",
    repeated_error: bool = False,
    edit_mode: bool = False,
) -> str:
    """Format a repair prompt: the full task prompt, then the failed script and its error.

    The task is repeated because each request is sent without conversation history.
    """
    missing_str = ""
    if missing_deliverables:
        missing_str = f"\nMissing Expected Deliverables in `{out_dir}`:\n" + "\n".join(
            f"- {f}" for f in missing_deliverables
        )

    repeat_str = ""
    if repeated_error:
        repeat_str = (
            "\nNOTE: this is the SAME error as the attempt before; your last fix did not work. "
            "Do not patch the same line again: re-read the data previews (index type, date "
            "frequency, keys) and change the approach.\n"
        )
    fix_instructions = (
        EDIT_INSTRUCTIONS
        if edit_mode
        else "Fix the bug and provide the entire corrected Python script (not a diff) inside a "
        "single ```python ... ``` code block."
    )
    return f"""{task_prompt}

---
{headline}

### PREVIOUS CODE:
```python
{previous_code}
```

### EXECUTION ERROR / OUTPUT:
```
{error_message}
```
{missing_str}
{repeat_str}
### INSTRUCTIONS:
1. Briefly diagnose the error (e.g. schema mismatch, file path issue, NaN/inf computation, or missing column).
2. Make sure all deliverables are written to OUTPUT_DIR (currently `{out_dir}`).
3. {fix_instructions}
"""


_NOTES_HEAD_CHARS = 3000
_NOTES_TAIL_CHARS = 9000


def _condense_notes(notes: str) -> str:
    notes = notes.replace("</think>", "").replace("```", "'''").strip()
    if len(notes) <= _NOTES_HEAD_CHARS + _NOTES_TAIL_CHARS:
        return notes
    return f"{notes[:_NOTES_HEAD_CHARS]}\n[... middle omitted ...]\n{notes[-_NOTES_TAIL_CHARS:]}"


def build_continuation_prompt(task_prompt: str, partial_code: str) -> str:
    """Ask for the rest of a script that was cut off at the output-token limit."""
    return f"""{task_prompt}

---
Your script for this task was cut off at the output-token limit. Everything written so far is
below; its last line is complete.

```python
{partial_code}
```

Continue the script from the next line. Output ONLY the remaining code in one ```python ... ```
block: do not repeat any line shown above and do not restart the script. Keep the rest compact.
"""


def build_no_code_prompt(truncated: bool, notes: str = "") -> str:
    """Ask again when the previous response contained no usable code block.

    ``notes`` is the previous (usually truncated) response; it is passed back as analysis so
    the retry spends its output budget on the script rather than on re-deriving the approach.
    """
    if truncated:
        reason = (
            "Your previous response was cut off at the output-token limit before a complete "
            "```python``` block was produced."
        )
    else:
        reason = "Your previous response did not contain a complete ```python ... ``` code block."
    notes_section = ""
    if notes.strip():
        notes_section = f"""
Your analysis from that response is below. Treat it as settled: do not re-derive or re-check it.

<previous_analysis>
{_condense_notes(notes)}
</previous_analysis>
"""
    return f"""{reason}
{notes_section}
Now think for only a few sentences, then write the complete script immediately. Keep it compact
(well under 200 lines, few comments). Put the complete script inside one ```python ... ``` block.
"""


def build_review_prompt(
    task_prompt: str,
    code: str,
    stdout: str,
    output_previews: str,
    findings: list[str],
    edit_mode: bool = False,
) -> str:
    """Ask the model to verify a clean run's outputs against the specification."""
    fix_form = (
        f"then the fixes as SEARCH/REPLACE edit blocks.\n\n{EDIT_INSTRUCTIONS}"
        if edit_mode
        else "and then the complete corrected script in one ```python ... ``` block. Your reply "
        "is capped at ~4,000 tokens, so keep the bullets terse and spend the budget on the script."
    )
    findings_str = "\n".join(f"- {f}" for f in findings) if findings else "- none"
    stdout_str = stdout.strip()[-1500:] or "(no output)"
    return f"""{task_prompt}

---
Your script below ran without errors and wrote the expected files. Before it is submitted,
verify the result against the TASK SPECIFICATION above. A hidden checker will test it.

### SCRIPT:
```python
{code}
```

### SCRIPT STDOUT (tail):
```
{stdout_str}
```

### AUTOMATIC FINDINGS ON THE OUTPUTS:
{findings_str}

### OUTPUT PREVIEWS:
{output_previews or "(none)"}

### REVIEW CHECKLIST:
1. Every required file, column, JSON key and nesting matches the specification exactly
   (names, order, types, units, percent vs decimal, sign and date conventions, row counts).
2. The method follows the specification's definitions, parameters and edge-case rules.
3. Values are plausible and satisfy the domain invariants (no-arbitrage bounds, sign rules,
   weights summing to 1, probabilities in [0, 1], internally consistent summary numbers).
4. Each automatic finding above is either justified by the specification or a bug.

Reply with `VERDICT: PASS` if you find no concrete problem. Otherwise reply with
`VERDICT: FAIL`, at most 8 one-line bullets naming the concrete problems, {fix_form}
Report only real, specific defects.
"""


def build_judge_prompt(task_prompt: str, diffs: list[str], code_a: str, code_b: str) -> str:
    """Ask which of two disagreeing candidate solutions follows the specification.

    Scripts are included only when given (the caller drops them if the request would not fit).
    """
    diffs_str = "\n".join(f"- {d}" for d in diffs) or "- (outputs differ)"
    scripts = ""
    if code_a and code_b:
        scripts = f"""
### CANDIDATE A SCRIPT:
```python
{code_a}
```

### CANDIDATE B SCRIPT:
```python
{code_b}
```
"""
    return f"""{task_prompt}

---
Two independently written scripts for this task both ran cleanly but produced DIFFERENT outputs.
Exactly one of them should be submitted.

### WHERE THE OUTPUTS DISAGREE:
{diffs_str}
{scripts}
Decide which candidate follows the TASK SPECIFICATION (definitions, conventions, units, edge
cases) more faithfully for the values that disagree. Think briefly, then end your reply with a
single line: `CHOICE: A` or `CHOICE: B`.
"""
