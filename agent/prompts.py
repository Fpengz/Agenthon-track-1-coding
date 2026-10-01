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


_SYSTEM_TEMPLATE = """You are an expert Quantitative Finance Engineer and Python Developer participating in Agenthon 2026 / QFBench 2.0 Track 1.

{objective}

CRITICAL OPERATING RULES:
1. Paths (IMPORTANT):
   - Task instructions refer to `/input/...` and `/output/...`. At run time these map to the
     directories in the environment variables TASK_DIR and OUTPUT_DIR. Always resolve paths as
     `pathlib.Path(os.environ["TASK_DIR"]) / "<path relative to /input>"` and
     `pathlib.Path(os.environ["OUTPUT_DIR"]) / "<file name>"`. Never hard-code `/input` or `/output`.
   - The script runs from a scratch directory; do not rely on the current working directory.
{output_rule}

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

{response_rules}""".replace("{versions}", _runtime_versions())

_SOLUTION_OBJECTIVE = "Your objective is to write robust, bug-free, self-contained Python code that solves the quantitative finance task described in the prompt and outputs the required deliverable files."
_SOLUTION_OUTPUT_RULE = """   - Write only the requested deliverables to OUTPUT_DIR. Do NOT write reward.json or
     pytest_report.json (the test harness generates these offline)."""
_SOLUTION_RESPONSE_RULES = """5. Response Budget and Formatting:
   - Your whole response (reasoning included) is capped at about 4,000 tokens. Keep reasoning
     short and spend the budget on the script. Keep the script compact (well under 200 lines),
     with few comments and no long docstrings.
   - Provide the complete, runnable Python script inside a single ```python ... ``` markdown block.
     Put nothing after the closing fence.
   - Do NOT output partial snippets or placeholders. The code will be executed directly via Python.
   - Use only ASCII characters in the code (no unicode math symbols such as the partial sign or sigma).
"""


def _system_prompt(objective: str, output_rule: str, response_rules: str) -> str:
    return (
        _SYSTEM_TEMPLATE.replace("{objective}", objective)
        .replace("{output_rule}", output_rule)
        .replace("{response_rules}", response_rules)
    )


SYSTEM_PROMPT = _system_prompt(_SOLUTION_OBJECTIVE, _SOLUTION_OUTPUT_RULE, _SOLUTION_RESPONSE_RULES)
TOOL_SYSTEM_PROMPT = _system_prompt(
    _SOLUTION_OBJECTIVE,
    _SOLUTION_OUTPUT_RULE,
    _SOLUTION_RESPONSE_RULES.replace(
        "Provide the complete, runnable Python script inside a single ```python ... ``` markdown block.",
        "For a solution, provide its complete runnable script in ```python ... ```. To request "
        "a local numerical probe first, provide only that standalone program in ```explore ... ```.",
    ),
)
PROBE_SYSTEM_PROMPT = _system_prompt(
    "Your current objective is to write one small executable numerical probe. Its measurements "
    "will inform a later solution or review; this request does not produce the final solution.",
    "   - OUTPUT_DIR is a fresh scratch copy of available deliverables. Read it when checking "
    "outputs, and print observations to stdout. Final deliverables are written in a later phase.",
    """5. Probe Budget and Formatting:
   - Reply only with one complete runnable program inside ```python ... ```.
   - Keep it under 60 lines and reasoning brief; the whole response is capped at 4,000 tokens.
   - Start with a comment naming the numerical question and its expected invariant.
   - Print compared values and their discrepancy. Use a tiny synthetic example or bounded data
     sample for initial probes; independently recompute and read an actual value for output probes.
   - Use ASCII code and installed local libraries. Put nothing after the closing fence.
""",
)


_MAX_PATH_MAP_LINES = 40

EXPLORE_INSTRUCTION = """5. Local numerical probes: up to {k} snippets across generation, repair and review.
   The agent requests initial and output probes in a separate phase, using ```python``` replies.
   In ordinary solution/repair/review requests, Python blocks are solutions or fixes; request
   any additional numerical measurement with one standalone ```explore``` block under 60 lines.
   Start with a comment naming a numerical question from this task and its expected invariant.
   Test a formula, sign/scale convention, date alignment or library behaviour on a tiny synthetic
   example or a bounded input sample. Print the competing values and numerical discrepancy.
   Use the input previews for schemas; spend the probe on a computation that resolves uncertainty.
   Keep input reads under TASK_DIR and use only local installed libraries. A probe's OUTPUT_DIR
   is a scratch copy of the latest deliverables, so you can read results during repair/review.
   Its stdout and errors come back before you write the solution or verdict. Resume the pending
   operation after observing them. The task specification determines which convention to use;
   a plausible number or a probe's PASS message alone does not establish correctness.
"""

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
    requirements_checklist: str = "",
    skills_summary: str = "",
    explore_steps: int = 0,
    validation_guidance: str = "",
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
{notes_str}{f"{chr(10)}### REQUIREMENTS CHECKLIST (extracted from the specification; verify each item):{chr(10)}{requirements_checklist}{chr(10)}" if requirements_checklist else ""}{reference_example}
### INSTRUCTIONS:
1. Write a complete, standalone Python script that reads inputs from TASK_DIR and writes the required deliverables to OUTPUT_DIR.
2. Ensure all column names, file formats, and data structures match the specification exactly.
3. Keep reasoning brief, then enclose your complete script inside one ```python ... ``` code block.
{STRUCTURE_INSTRUCTION if structured else ""}{EXPLORE_INSTRUCTION.format(k=explore_steps) if explore_steps else ""}{f"{chr(10)}{skills_summary}{chr(10)}" if skills_summary else ""}{f"{chr(10)}{validation_guidance}{chr(10)}" if validation_guidance else ""}"""


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


def build_review_retry_prompt(review_prompt: str, previous_response: str) -> str:
    """Finish an incomplete review while retaining its script and output evidence."""
    return f"""{review_prompt}

---
Your previous review did not provide a complete verdict. The clean outputs are still saved.
Finish the review and put `VERDICT: PASS` or `VERDICT: FAIL` FIRST in your final answer.
For FAIL, include the concrete defects and the corrected script or edits requested above.
Keep reasoning brief; do not spend the response budget repeating the analysis below.

### PREVIOUS INCOMPLETE REVIEW (provisional, check against the specification):
{_condense_notes(previous_response)}
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


def build_verify_prompt(task_prompt: str, code: str, output_previews: str) -> str:
    """Ask for an executable, independent verification script for a clean run's outputs."""
    return f"""{task_prompt}

---
The script below ran cleanly and wrote the deliverables. Write a SEPARATE verification script
that checks those outputs against the TASK SPECIFICATION, so that mistakes are caught by
execution rather than by reading.

### SCRIPT THAT PRODUCED THE OUTPUTS:
```python
{code}
```

### OUTPUT PREVIEWS:
{output_previews or "(none)"}

### VERIFIER REQUIREMENTS:
1. Read inputs from TASK_DIR and the deliverables from OUTPUT_DIR (environment variables). Do not
   write or modify any file.
2. Where practical, recompute 2-4 key values INDEPENDENTLY (a different formula, method or
   aggregation than the script uses) and compare within a sensible tolerance.
3. Check what the specification states: required files, columns and keys, row counts, units and
   conventions (percent vs decimal, signs, dates), ranges and invariants.
   If the prompt lists `agent_skills` reference implementations, use them as the independent
   computation (they are tested and do not share the script's assumptions).
4. Print one line per check, starting with `PASS:` or `FAIL:` and, for failures, the expected vs
   actual value. Exit with status 1 if any check fails, 0 otherwise.
   Evaluate the checks before printing PASS. Catch individual assertion failures, print FAIL
   with the assertion's values, and continue with the remaining checks.
5. Keep it compact and put it in one ```python ... ``` block.
"""


def build_spec_tests_prompt(task_prompt: str) -> str:
    """Write tests before the solution exists, with no solution or output context."""
    return f"""{task_prompt}

---
You are writing tests for DELIVERABLES that another program will produce.
The test program must READ those deliverables and compare their values with independent
calculations. Checking the supplied inputs alone cannot establish whether the solution works.

Your role for this request is TEST AUTHOR. Another request independently writes the solution.
Write a standalone Python test program derived ONLY from the TASK SPECIFICATION and supplied
input information above. The solution and its outputs are unavailable to you.

### SPEC-FIRST TEST REQUIREMENTS:
1. Start by setting out = pathlib.Path(os.environ["OUTPUT_DIR"]) and loading the required
   deliverable files from out. These files will exist when your tests execute. Read original
   inputs from TASK_DIR as needed for independent expected values, using the mapping above.
   Treat all files as read-only. Return tests, rather than a program that writes deliverables.
2. Independently compute TWO representative numerical results from the supplied input files.
   Use the specification's exact formulas, filters, timing, units and conventions. Check values
   against the LOADED DELIVERABLE values by their stated row identifiers or keys. Prefer small
   direct calculations over implementing the entire solution. If agent_skills helpers are
   listed, use them where conventions match.
3. Also check explicitly required schema and financial invariants. Test only requirements the
   specification establishes. Where a numerical convention is ambiguous, test a stated
   invariant instead of inventing an exact expected value. Use sensible numerical tolerances;
   stochastic estimates need sampling error tolerances, not equality to one seeded draw.
4. Print a nonempty PASS: or FAIL: line for every executed check. A failure must include the
   requirement, expected value and actual value. Catch assertion failures separately so the
   remaining checks run. Exit 1 if any check fails and 0 otherwise. Print PASS only after
   executing a comparison. Missing required files or fields are failures.
5. Keep the program UNDER 80 LINES. Prioritize the two numerical comparisons and essential
   output schema checks. Return one complete ```python ... ``` block with no preamble.
"""


def build_verifier_repair_prompt(verifier_prompt: str, verifier: str, feedback: str) -> str:
    """Repair the verification program rather than an untested solution."""
    return f"""{verifier_prompt}

---
The VERIFICATION SCRIPT could not complete its checks. Repair that verifier only. This error
does not establish that the solution is wrong. Read the saved outputs from OUTPUT_DIR and
inputs from TASK_DIR, do not write deliverables, and print PASS: or FAIL: for every check.

### PREVIOUS VERIFICATION SCRIPT:
```python
{verifier or "# No complete verification script was returned."}
```

### VERIFIER FEEDBACK:
{feedback[-4500:]}

Return the complete repaired VERIFICATION SCRIPT in one ```python ... ``` block.
"""


VERIFY_REPAIR_HEADLINE = (
    "Your script ran cleanly, but an independent verification script reported the failures "
    "below. The verifier may itself be wrong: if your script follows the specification, return "
    "it UNCHANGED (the same complete script); otherwise fix it."
)


def build_plan_prompt(task_prompt: str) -> str:
    """Ask for a requirements checklist extracted from the specification (no code)."""
    return f"""{task_prompt}

---
Do NOT write code yet. Extract a numbered checklist of every requirement in the TASK
SPECIFICATION that a checker could test: each deliverable file with its exact columns or keys
and their order, formulas and their parameters, units and conventions (percent vs decimal,
annualisation, sign, day count, date alignment), filters and edge cases, sorting and row counts.
Quote exact names. One short line per item, at most 30 items, no preamble.
"""


def build_exploration_section(log: list[tuple[str, str]]) -> str:
    """Executed observations retained in every task-prompt variant and follow-up."""
    parts = ["\n### EXPLORATION SO FAR:"]
    for i, (snippet, output) in enumerate(log, 1):
        parts.append(f"#### Snippet {i}\n```python\n{snippet}\n```\nOutput:\n```\n{output}\n```")
    parts.append(
        "\nUse these executed observations with their recorded solution turn. "
        "Output observations from an earlier solution describe that version; "
        "check new outputs again after changing the script."
    )
    return "\n".join(parts)


def build_probe_instruction(steps_left: int, phase: str) -> str:
    """Tool availability for the current operation, separate from persistent evidence."""
    if phase == "continue":
        return "\nContinue the pending solution script; probes are unavailable during continuation."
    if not steps_left:
        return (
            f"\nNo exploration left: finish the pending {phase} in its requested format "
            "(complete solution, repair edits, or review verdict)."
        )
    return (
        f"\nLOCAL PROBE AVAILABLE: {steps_left} remaining. To measure a specific numerical "
        f"uncertainty before finishing this {phase}, reply ONLY with one complete ```explore``` "
        "block (under 60 lines). OUTPUT_DIR is a fresh scratch copy of current deliverables. "
        "Its results return to this same operation. Otherwise finish in the format requested above."
    )


def build_probe_request(phase: str) -> str:
    """An explicit request phase; Python replies are measurements, never solution scripts."""
    question = (
        "Independently recompute one important requested value from the specification and "
        "task inputs. Read its actual deliverable value using OUTPUT_DIR and print both values "
        "and their numerical discrepancy. The solution code is provisional; choose the "
        "calculation from the specification."
        if phase == "review"
        else "Resolve one numerical uncertainty before implementing the solution: compare "
        "formulas, sign/scale conventions, date alignment or library behaviour on a tiny "
        "synthetic example or bounded input sample. Print the values and discrepancy; use "
        "the supplied previews for schema information."
    )
    return f"""

### CURRENT PHASE: NUMERICAL PROBE BEFORE {phase.upper()}
The material above describes the pending {phase}; this request asks only for a diagnostic program.
{question}
OUTPUT_DIR is a fresh scratch copy of available outputs. Return one complete ```python``` program
under 60 lines. Its stdout/errors return to the pending {phase} before a solution or verdict.
"""
