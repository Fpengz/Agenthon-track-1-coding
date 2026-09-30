"""Tests for the base agent implementation."""

import pathlib
import re
import tempfile

from agent.client import ChatResult
from agent.executor import (
    apply_edits,
    extract_partial_code,
    extract_python_code,
    join_continuation,
    run_code,
)
from agent.inputs import build_input_previews
from agent.loop import AgentSolver, container_path_map
from agent.review import output_diagnostics, parse_verdict


def test_extract_python_code():
    markdown = """Here is the solution:
```python
import numpy as np
print("hello")
```
Hope this helps!"""
    extracted = extract_python_code(markdown)
    assert extracted == 'import numpy as np\nprint("hello")'


def test_extract_python_code_skips_inline_reasoning():
    response = "Draft: ```python\nold()\n```\n</think>\n\n```python\nprint(1)\n```"
    assert extract_python_code(response) == "print(1)"


def test_extract_python_code_rejects_truncated_block():
    # Output cut off at the token limit: never execute prose or an unclosed block.
    assert extract_python_code("We need dV/dt...\n</think>\n```python\nx = (") == ""
    assert extract_python_code("Reasoning cut off mid-thought") == ""


def test_executor_run():
    with tempfile.TemporaryDirectory() as tmp_dir:
        work = pathlib.Path(tmp_dir)
        code = 'print("hello executor")'
        result = run_code(code, work / "work", work / "out", task_dir=work)
        assert result.success
        assert "hello executor" in result.stdout


class DummyMockClient:
    def __init__(self):
        self.request_count = 0

    def chat(self, messages, **kwargs):
        self.request_count += 1
        return ChatResult(content=CODE_RESPONSE, finish_reason="stop")


CODE_RESPONSE = """```python
import pandas as pd
import pathlib
import os

out = pathlib.Path(os.environ.get("OUTPUT_DIR", "."))
df = pd.DataFrame({"option_id": [1], "price": [10.0]})
df.to_parquet(out / "results.parquet")
```"""


def test_agent_solver_mock():
    with tempfile.TemporaryDirectory() as tmp_out:
        task_dir = pathlib.Path("units/t1-EXAMPLE-bs-greeks-pde")
        out_dir = pathlib.Path(tmp_out)
        solver = AgentSolver(
            task_dir=task_dir,
            out_dir=out_dir,
            client=DummyMockClient(),
        )
        success = solver.run()
        assert success
        assert (out_dir / "results.parquet").exists()


def test_container_path_map_resolves_renamed_copies(tmp_path):
    env = tmp_path / "environment"
    (env / "data" / "sub").mkdir(parents=True)
    (env / "data" / "raw.pqt").write_text("x")
    (env / "data" / "sub" / "a.csv").write_text("x")
    (env / "Dockerfile").write_text(
        "FROM base\n"
        "COPY data/raw.pqt /app/data/panel.parquet\n"
        "COPY data/raw.pqt /app/\n"
        "COPY data/ /app/data/\n"
    )
    mapping = dict(container_path_map(tmp_path))
    assert mapping["/app/data/panel.parquet"] == "environment/data/raw.pqt"
    assert mapping["/app/raw.pqt"] == "environment/data/raw.pqt"
    assert mapping["/app/data/sub/a.csv"] == "environment/data/sub/a.csv"


def test_discover_inputs_skips_answers_and_keeps_any_environment_file(tmp_path):
    (tmp_path / "instruction.md").write_text("Write `/output/chart.png` and /output/r.json")
    (tmp_path / "environment" / "data").mkdir(parents=True)
    (tmp_path / "environment" / "data" / "panel.pqt").write_text("x")
    (tmp_path / "environment" / "Dockerfile").write_text("FROM base\n")
    (tmp_path / "checks" / "reference_data").mkdir(parents=True)
    (tmp_path / "checks" / "reference_data" / "expected.json").write_text("{}")
    solver = AgentSolver(task_dir=tmp_path, out_dir=tmp_path / "out", client=DummyMockClient())
    inputs = [p.relative_to(tmp_path).as_posix() for p in solver.discover_inputs()]
    assert inputs == ["environment/data/panel.pqt"]
    text = solver.find_instruction()
    assert solver.detect_expected_deliverables(text) == ["chart.png", "r.json"]


def test_extract_partial_code_keeps_complete_lines_of_cut_off_block():
    response = "</think>\n```python\nimport os\nx = 1\ny = (2 +"
    assert extract_partial_code(response) == "import os\nx = 1"
    assert extract_partial_code("```python\nprint(1)\n```") == ""  # closed: nothing cut off
    assert extract_partial_code("no code at all") == ""


def test_join_continuation_appends_or_accepts_a_restart():
    assert join_continuation("import os\nx = 1", "y = 2") == "import os\nx = 1\ny = 2"
    restart = "import os\nx = 1\ny = 2"
    assert join_continuation("import os\nx = 1", restart) == restart


def test_input_previews_show_columns_and_json_keys(tmp_path):
    data = tmp_path / "environment" / "data"
    data.mkdir(parents=True)
    (data / "px.csv").write_text("date,close\n2024-01-02,10\n2024-01-03,11\n")
    (data / "cfg.json").write_text('{"bucket_ms": 500, "levels": [1, 2, 3, 4]}')
    text = build_input_previews(tmp_path, sorted(data.iterdir()))
    assert "date,close" in text and "2 data rows" in text
    assert '"bucket_ms": 500' in text and "4 items total" in text


def test_continuation_block_keeps_leading_indentation():
    piece = extract_python_code("```python\n    return x\n\nmain()\n```")
    assert piece == "    return x\n\nmain()"
    assert join_continuation("def f(x):", piece) == "def f(x):\n    return x\n\nmain()"


class ScriptedClient:
    """Returns canned responses in order; records how many requests were made."""

    max_requests = 25

    def __init__(self, *responses):
        self.responses = list(responses)
        self.request_count = 0

    def chat(self, messages, **kwargs):
        self.request_count += 1
        self.prompts = getattr(self, "prompts", []) + [messages[-1]["content"]]
        return ChatResult(content=self.responses.pop(0), finish_reason="stop")


WRITE_OK = (
    "```python\nimport os, pathlib\n"
    "out = pathlib.Path(os.environ['OUTPUT_DIR'])\n"
    "(out / 'r.json').write_text('{\"price\": 1.5}')\n```"
)
CRASH = "VERDICT: FAIL\n- price is wrong\n```python\nraise SystemExit(1)\n```"


def _solver(tmp_path, client):
    (tmp_path / "task").mkdir()
    (tmp_path / "task" / "instruction.md").write_text("Write /output/r.json")
    return AgentSolver(task_dir=tmp_path / "task", out_dir=tmp_path / "out", client=client)


def test_review_pass_accepts_after_one_review(tmp_path):
    client = ScriptedClient(WRITE_OK, "Looks right.\nVERDICT: PASS")
    solver = _solver(tmp_path, client)
    assert solver.run()
    assert client.request_count == 2
    assert (tmp_path / "out" / "r.json").read_text() == '{"price": 1.5}'


def test_failed_review_fix_restores_last_clean_outputs(tmp_path):
    # Every "fix" crashes, so the loop runs out of attempts and must restore the clean run.
    client = ScriptedClient(WRITE_OK, CRASH, *[CRASH] * 20)
    solver = _solver(tmp_path, client)
    solver.max_retries = 3
    assert solver.run()
    assert (tmp_path / "out" / "r.json").read_text() == '{"price": 1.5}'


def test_parse_verdict_and_output_diagnostics(tmp_path):
    assert parse_verdict("ok\n**VERDICT: PASS**") == "PASS"
    assert parse_verdict("VERDICT: FAIL\n...") == "FAIL"
    assert parse_verdict("no verdict") is None
    (tmp_path / "t.csv").write_text("a,b\n1,\n2,inf\n")
    (tmp_path / "r.json").write_text('{"x": NaN, "y": null, "z": []}')
    found = "\n".join(output_diagnostics(tmp_path))
    assert "'b' has 1/2 missing" in found and "'b' has 1 infinite" in found
    assert (
        "NaN/Infinity" in found
        and "r.json.y is null" in found
        and "r.json.z is an empty list" in found
    )


def test_metrics_report_uses_fixed_denominator_and_official_labels(tmp_path):
    from agent.metrics import assess_unit, build_report, format_report

    out = tmp_path / "out"
    out.mkdir()
    (out / "r.json").write_text('{"note": "abc-canary-123"}')
    assessed = assess_unit(tmp_path, out, timed_out=False, reward=1.0, registry={"abc-canary-123"})
    assert assessed["gates"]["g2_cutoff_resource"] is False
    assert assessed["failure_label"] == "shared.contamination.canary_emitted"

    def row(unit, status, label, reqs):
        return {
            "unit": unit, "status": status, "failure_label": label, "model_requests": reqs,
            "agent_seconds": 60.0, "tests_passed": 1, "tests_total": 2,
            "gates": {"g3_domain_semantics": status == "passed"},
            "category": "pricing", "difficulty": "hard",
        }  # fmt: skip

    rows = [
        row("a", "passed", None, 2),
        row("b", "failed", "t1.wrong_numeric", 1),
        row("c", "timeout", "shared.resource.timeout", 0),
    ]
    report = build_report(rows, checked=True)
    assert report["official"]["pass_at_1"] == round(1 / 3, 4)  # timeouts stay in the denominator
    assert report["failure_labels"] == {"t1.wrong_numeric": 1, "shared.resource.timeout": 1}
    assert report["diagnostics"]["house_requests"]["units_without_house_request"] == ["c"]
    assert "pass@1 (official metric): 0.333" in format_report(report)


def test_review_fail_without_script_triggers_a_repair_turn(tmp_path):
    fixed = WRITE_OK.replace("1.5", "2.5")
    client = ScriptedClient(
        WRITE_OK, "VERDICT: FAIL\n- price uses the wrong day count", fixed, "VERDICT: PASS"
    )
    solver = _solver(tmp_path, client)
    assert solver.run()
    assert client.request_count == 4
    assert (tmp_path / "out" / "r.json").read_text() == '{"price": 2.5}'


def test_unchanged_failing_script_is_not_rerun(tmp_path):
    crash = "```python\nraise SystemExit(1)\n```"
    client = ScriptedClient(crash, crash, WRITE_OK, "VERDICT: PASS")
    solver = _solver(tmp_path, client)
    assert solver.run()
    assert client.request_count == 4  # the repeat is re-asked, not executed


def test_partial_code_is_trimmed_to_complete_statements():
    response = "```python\nimport os\nx = 1\nm = m[\n    ['a', 'b',\n"
    assert extract_partial_code(response) == "import os\nx = 1"


def test_apply_edits_exact_tolerant_and_unmatched():
    code = "a = 1\n    b = 2   \nc = 3"
    # Multi-line SEARCH without the trailing spaces: no exact substring, so the tolerant path.
    edit = "<<<<<<< SEARCH\n    b = 2\nc = 3\n=======\n    b = 20\nc = 3\n>>>>>>> REPLACE"
    assert apply_edits(code, edit) == ("a = 1\n    b = 20\nc = 3", [])
    missing = "<<<<<<< SEARCH\nzzz\n=======\nq\n>>>>>>> REPLACE"
    assert apply_edits(code, missing) == (
        None,
        ["SEARCH starting 'zzz': line not in the script: 'zzz'"],
    )
    assert apply_edits(code, "```python\nprint(1)\n```") == (None, [])


def test_long_script_is_repaired_with_an_edit_block(tmp_path):
    filler = "".join(f"v{i} = {i}\n" for i in range(90))
    broken = (
        "import os, pathlib\n" + filler + "out = pathlib.Path(os.environ['OUTPUT_DIR'])\n"
        "value = {'price': 1.5}['cost']\n"
        "(out / 'r.json').write_text('{\"price\": 1.5}')\n"
    )
    fix = (
        "The key is wrong.\n<<<<<<< SEARCH\nvalue = {'price': 1.5}['cost']\n=======\n"
        "value = {'price': 1.5}['price']\n>>>>>>> REPLACE\n"
    )
    client = ScriptedClient(f"```python\n{broken}```", fix, "VERDICT: PASS")
    solver = _solver(tmp_path, client)
    assert solver.run()
    assert client.request_count == 3
    assert (tmp_path / "out" / "r.json").read_text() == '{"price": 1.5}'


def test_apply_edits_reindents_and_reports_ambiguity():
    code = "def f():\n    if x:\n        y = 1\n    return y\nz = 0\nz = 0"
    wrong_indent = "<<<<<<< SEARCH\nif x:\n    y = 1\n=======\nif x:\n    y = 2\n>>>>>>> REPLACE"
    assert apply_edits(code, wrong_indent)[0] == (
        "def f():\n    if x:\n        y = 2\n    return y\nz = 0\nz = 0"
    )
    ambiguous = "<<<<<<< SEARCH\nz = 0\n=======\nz = 1\n>>>>>>> REPLACE"
    new, problems = apply_edits(code, ambiguous)
    assert new is None and "matches 2 places" in problems[0]


def _example(unit_id, instruction, solution="print('ok')"):
    from agent.examples import instruction_hash

    return {
        "unit_id": unit_id,
        "instruction_sha256": instruction_hash(instruction),
        "instruction": instruction,
        "tags": [],
        "solution": solution,
    }


def test_select_example_never_uses_the_tasks_own_or_duplicate_solution(tmp_path):
    from agent.examples import select_example

    task = "Price European options with a Black-Scholes finite difference PDE grid and Greeks."
    (tmp_path / "card.toml").write_text('[task]\nid = "t1-me"\n')
    other = "Compute bond duration convexity for a zero coupon curve and immunization weights."
    library = [
        _example("t1-me", "totally different text"),  # same unit id
        _example("t1-copy", task),  # same instruction under another id
        _example("t1-near", task + " Output results.parquet."),  # near-duplicate re-issue
        _example("t1-bond", other),  # permitted but irrelevant
    ]
    assert select_example(tmp_path, task, library) is None
    related = "Black-Scholes Greeks: delta gamma vega theta for option chains via PDE grids."
    chosen = select_example(tmp_path, task, library + [_example("t1-greeks", related)])
    assert chosen is not None and chosen["unit_id"] == "t1-greeks"


def test_domain_notes_pick_the_matching_category(tmp_path, monkeypatch):
    from agent.knowledge import domain_notes

    monkeypatch.setenv("AGENT_DOMAIN_NOTES", "1")
    (tmp_path / "card.toml").write_text(
        '[metadata]\ncategory = "fixed-income"\ntags = ["yield-curve"]\n'
    )
    notes = domain_notes(tmp_path, "Bootstrap a zero coupon yield curve and compute bond duration.")
    assert "`fixed-income`" in notes and "Common mistakes" in notes


def test_fit_to_context_drops_optional_parts_then_shrinks_output(monkeypatch):
    from agent import loop
    from agent.prompts import SYSTEM_PROMPT

    base = int(len(SYSTEM_PROMPT) / loop.CHARS_PER_TOKEN)
    monkeypatch.setattr(loop, "CONTEXT_TOKENS", base + 4000 + 256 + 400)  # ~1000 chars of room
    full, lean = "TASK " + "e" * 3000, "TASK"
    variants = [full, full[:2000], lean]
    repair = full + "\n---\nfix this"

    fitted = loop.fit_to_context(repair, full, variants)
    assert fitted == (lean + "\n---\nfix this", lean, 4000)  # leanest variant, suffix kept
    assert loop.fit_to_context("x" * 500, "x", ["x"]) == ("x" * 500, "x", 4000)  # already fits

    monkeypatch.setattr(loop, "CONTEXT_TOKENS", base + 256 + 2000)
    user, _, max_tokens = loop.fit_to_context("TASK" + "y" * 400, "TASK", ["TASK"])
    assert 1024 <= max_tokens < 4000  # no leaner variant left: output cap shrinks

    monkeypatch.setattr(loop, "CONTEXT_TOKENS", base + 256 + 500)
    assert loop.fit_to_context("TASK" + "y" * 4000, "TASK", ["TASK"]) is None


def test_time_budget_stops_before_starting_an_attempt(tmp_path):
    import time

    client = ScriptedClient(WRITE_OK)
    solver = _solver(tmp_path, client)
    solver.deadline = time.monotonic() + 5  # less than MIN_ATTEMPT_SEC left
    assert not solver.run()
    assert client.request_count == 0


def test_unexpected_error_after_clean_run_keeps_accepted_outputs(tmp_path, monkeypatch):
    from agent import loop

    def boom(**kwargs):
        raise RuntimeError("bug while building the review prompt")

    monkeypatch.setattr(loop, "build_review_prompt", boom)
    solver = _solver(tmp_path, ScriptedClient(WRITE_OK))
    assert solver.run()
    assert (tmp_path / "out" / "r.json").read_text() == '{"price": 1.5}'


def test_experiment_registry_roundtrip_and_compare(tmp_path):
    from agent import experiments

    registry = tmp_path / "registry.jsonl"

    def record(run_id, statuses):
        summary = {
            "units": len(statuses),
            "units_planned": len(statuses),
            "metrics": {"official": {"pass_at_1": None, "passed": 0}, "diagnostics": {}},
            "results": [{"unit": u, "status": s} for u, s in statuses.items()],
        }
        manifest = {"run_id": run_id, "name": run_id, "note": "", "git": {"commit": "abc"}}
        experiments.append_registry(experiments.registry_entry(manifest, summary), registry)

    record("a", {"u1": "passed", "u2": "failed", "u3": "passed"})
    record("b", {"u1": "passed", "u2": "passed", "u3": "failed", "u4": "passed"})
    entries = [experiments.find_run(r, registry) for r in ("a", "b")]
    report = experiments.compare(entries)
    assert "3 units shared" in report  # u4 is not in run a
    assert "+ u2" in report and "- u3" in report
    assert "passed in any run: 3; in every run: 1" in report
    assert experiments.make_run_id("Notes On!").endswith("-notes-on")


def test_domain_notes_switch(tmp_path, monkeypatch):
    from agent.knowledge import domain_notes

    (tmp_path / "card.toml").write_text('[metadata]\ncategory = "fixed-income"\n')
    monkeypatch.delenv("AGENT_DOMAIN_NOTES", raising=False)  # off by default
    assert domain_notes(tmp_path, "Bootstrap a zero coupon yield curve.") == ""
    monkeypatch.setenv("AGENT_DOMAIN_NOTES", "0")
    assert domain_notes(tmp_path, "Bootstrap a zero coupon yield curve.") == ""
    monkeypatch.setenv("AGENT_DOMAIN_NOTES", "1")
    assert "fixed-income" in domain_notes(tmp_path, "Bootstrap a zero coupon yield curve.")


def test_compare_outputs_tolerance_and_diffs(tmp_path):
    from agent.consensus import compare_outputs

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "t.csv").write_text("id,v\nx,1.0000\ny,2.0\n")
    (b / "t.csv").write_text("id,v\nx,1.0001\ny,2.0\n")  # within tolerance
    (a / "r.json").write_text('{"price": 1.5, "n": 3}')
    (b / "r.json").write_text('{"price": 1.5, "n": 3}')
    assert compare_outputs(a, b).agree
    (b / "r.json").write_text('{"price": 2.5, "n": 3}')
    disagreement = compare_outputs(a, b)
    assert not disagreement.agree
    assert any("r.json:price" in d for d in disagreement.diffs)


def _writer(price):
    return WRITE_OK.replace("1.5", str(price))


SPEC_INSTRUCTION = """# Task
### `/output/prices.csv`
| Column | Type |
|---|---|
| `id` | str |
| `price` | float |

### File 2: stats.csv
Columns: `mean, std`

### `/output/summary.json`
```json
{"n": <int>, "fit": {"alpha": <float>, "beta": <float>}}
```
"""


def test_parse_spec_formats_and_missing_problems(tmp_path):
    from agent.spec import parse_spec, spec_problems

    specs = {
        s.name: s for s in parse_spec(SPEC_INSTRUCTION, ["prices.csv", "stats.csv", "summary.json"])
    }
    assert specs["prices.csv"].columns == ["id", "price"]
    assert specs["stats.csv"].columns == ["mean", "std"]
    assert specs["summary.json"].json_keys == ["n", "fit", "fit.alpha", "fit.beta"]
    (tmp_path / "prices.csv").write_text("id,price,extra\na,1,2\n")  # extra columns are fine
    (tmp_path / "stats.csv").write_text("mean\n1\n")
    (tmp_path / "summary.json").write_text('{"n": 3, "fit": {"alpha": 1}}')
    problems = "\n".join(spec_problems(list(specs.values()), tmp_path))
    assert "prices.csv" not in problems
    assert "stats.csv: missing required column(s) ['std']" in problems
    assert "summary.json: missing required key(s) ['fit.beta']" in problems


def test_spec_check_turns_a_clean_but_incomplete_run_into_one_repair(tmp_path, monkeypatch):
    from agent import loop

    monkeypatch.setattr(loop, "SPEC_CHECKS", True)
    (tmp_path / "task").mkdir()
    (tmp_path / "task" / "instruction.md").write_text(
        '### `/output/r.json`\n```json\n{"price": <float>, "n": <int>}\n```\n'
    )
    complete = WRITE_OK.replace('{"price": 1.5}', '{"price": 1.5, "n": 2}')
    client = ScriptedClient(WRITE_OK, complete, "VERDICT: PASS")
    solver = AgentSolver(task_dir=tmp_path / "task", out_dir=tmp_path / "out", client=client)
    assert solver.run()
    assert client.request_count == 3  # first script lacked "n" -> spec repair -> review
    assert (tmp_path / "out" / "r.json").read_text() == '{"price": 1.5, "n": 2}'


def test_structured_instruction_only_when_enabled():
    from agent.prompts import STRUCTURE_INSTRUCTION, build_initial_prompt

    args = dict(instruction_text="t", task_dir="/i", out_dir="/o", discovered_files=[])
    assert STRUCTURE_INSTRUCTION not in build_initial_prompt(**args)
    assert STRUCTURE_INSTRUCTION in build_initial_prompt(**args, structured=True)


CANARY = "103019c8-8341-49bc-bd88-f749ce8ca90b"


def test_sanitize_instruction_removes_every_canary_form(tmp_path):
    from agent.context import card_facts, sanitize_instruction, task_canaries

    (tmp_path / "card.toml").write_text(
        f'[contamination]\ncanary_guid = "{CANARY}"\n[metadata]\ncategory = "fx"\n'
        "[agent]\ntimeout_sec = 1800.0\n[environment]\ncpus = 16\n"
    )
    text = (
        f"<!-- BENCHMARK DATA SHOULD NEVER APPEAR -->\n# finance-bench-canary GUID {CANARY}\n"
        f"# Task\nUse id {CANARY.upper()} nowhere.\n"
    )
    canaries = task_canaries(tmp_path, text)
    clean = sanitize_instruction(text, canaries)
    assert CANARY not in clean.lower() and "canary" not in clean.lower()
    assert clean.startswith("# Task")
    facts = card_facts(tmp_path)
    assert "Category: fx" in facts and "1800 s" in facts and CANARY not in facts


def test_leaked_canary_is_never_accepted(tmp_path):
    (tmp_path / "task").mkdir()
    (tmp_path / "task" / "instruction.md").write_text(
        f"<!-- canary GUID {CANARY} -->\nWrite /output/r.json"
    )
    leaking = WRITE_OK.replace('{"price": 1.5}', f'{{"price": 1.5, "id": "{CANARY}"}}')
    client = ScriptedClient(leaking, WRITE_OK, "VERDICT: PASS")
    solver = AgentSolver(task_dir=tmp_path / "task", out_dir=tmp_path / "out", client=client)
    assert solver.run()
    assert client.request_count == 3  # leak -> repair -> clean run -> review
    assert CANARY not in (tmp_path / "out" / "r.json").read_text()


def test_error_context_hook_reports_columns_and_failure_type(tmp_path):
    code = (
        "import pandas as pd\n"
        "prices = pd.DataFrame({'Ticker Symbol': ['A'], 'close': [1.0]})\n"
        "x = prices['Ticker']\n"
    )
    result = run_code(code, tmp_path / "w", tmp_path / "o", task_dir=tmp_path, timeout_sec=60)
    assert result.failure_kind == "missing key or column (KeyError)"
    assert "[agent] failing line 3: x = prices['Ticker']" in result.stderr
    assert "columns=['Ticker Symbol', 'close']" in result.stderr
    assert "sitecustomize.py" not in [p.name for p in (tmp_path / "o").iterdir()]


def test_spec_value_checks_and_false_positive_guards(tmp_path):
    from agent.spec import parse_spec, spec_problems

    instruction = """# Deliverables
- **`fits.csv`**: Columns: `stock1`, `copula`, `aic`. One row per pair (3 rows total). `copula` values: `gaussian`, `clayton`.
- **`other.csv`**: Columns: `a`, `b`.

### `/output/prices.csv`
| Column | Type |
|---|---|
| `id` | str |
| `price` | float64 |

### `/output/summary.json`
```json
{"n": <int>, "label": <str>, "weights": {"<sector>": <float>, ...}}
```
"""
    specs = {s.name: s for s in parse_spec(instruction, ["fits.csv", "prices.csv", "summary.json"])}
    assert specs["fits.csv"].columns == ["stock1", "copula", "aic"]  # not the copula values
    assert specs["fits.csv"].row_count == 3
    assert specs["prices.csv"].numeric_columns == ["price"]
    assert specs["summary.json"].json_keys == ["n", "label", "weights"]  # no dynamic keys
    assert specs["summary.json"].numeric_keys == ["n"]
    (tmp_path / "fits.csv").write_text("stock1,copula,aic\nA,gaussian,1\nB,clayton,2\n")
    (tmp_path / "prices.csv").write_text("id,price\na,1.5\nb,abc\n")
    (tmp_path / "summary.json").write_text('{"n": "3", "label": "x", "weights": {"tech": 0.4}}')
    problems = "\n".join(spec_problems(list(specs.values()), tmp_path))
    assert "fits.csv: has 2 rows; the specification says 3 rows" in problems
    assert "prices.csv: column 'price' must be numeric" in problems and "'abc'" in problems
    assert "summary.json: key 'n' must be a number" in problems
    assert "weights" not in problems


def test_low_effort_falls_back_to_off_when_not_honoured(monkeypatch):
    from agent.client import HouseModelClient

    monkeypatch.delenv("HOUSE_REASONING", raising=False)
    client = HouseModelClient(base_url="http://127.0.0.1:9", model_name="x")
    assert client.reasoning_mode == "low"  # the default
    client._check_low_effort("Brief plan.\n</think>\n```python\nprint(1)\n```", "stop")
    assert client.reasoning_mode == "low"  # honoured: short reasoning
    client._check_low_effort("thinking " * 1000 + "</think>\n```python\nx\n```", "stop")
    assert client.reasoning_mode == "off"  # long reasoning: not honoured
    client.reasoning_mode = "low"
    client._check_low_effort("still thinking about the approach", "length")
    assert client.reasoning_mode == "off"  # cut off before any code


class RoutedClient:
    """Thread-safe test client: responses per consensus candidate, routed by the staging
    directory (out-A, out-B, ...) named in the prompt; anything else goes to the judge."""

    max_requests = 25

    def __init__(self, judge=(), **per_candidate):
        import threading

        self.queues = {label: list(items) for label, items in per_candidate.items()}
        self.queues["judge"] = list(judge)
        self.request_count = 0
        self.lock = threading.Lock()

    def chat(self, messages, **kwargs):
        match = re.search(r"out-([A-E])\b", messages[-1]["content"])
        key = match.group(1) if match else "judge"
        with self.lock:
            self.request_count += 1
            queue = self.queues.get(key) or []
            content = queue.pop(0) if queue else "no more scripted replies"
        return ChatResult(content=content, finish_reason="stop")


def _consensus(tmp_path, monkeypatch, n, **client_args):
    from agent import loop

    monkeypatch.setattr(loop, "CANDIDATES", n)
    client = RoutedClient(**client_args)
    return _solver(tmp_path, client), client


def _out(tmp_path):
    return (tmp_path / "out" / "r.json").read_text()


def test_parallel_consensus_accepts_agreeing_candidates(tmp_path, monkeypatch):
    solver, _ = _consensus(
        tmp_path, monkeypatch, 3,
        A=[_writer(1.5), "VERDICT: PASS"], B=[_writer(1.5)], C=[_writer(1.5)],
    )  # fmt: skip
    assert solver.run()
    assert _out(tmp_path) == '{"price": 1.5}'


def test_parallel_consensus_agreeing_pair_overrules_candidate_a(tmp_path, monkeypatch):
    solver, _ = _consensus(
        tmp_path, monkeypatch, 3,
        A=[_writer(1.5), "VERDICT: PASS"], B=[_writer(2.5)], C=[_writer(2.5)],
    )  # fmt: skip
    assert solver.run()
    assert _out(tmp_path) == '{"price": 2.5}'  # B and C agree


def test_parallel_consensus_judge_settles_two_candidates(tmp_path, monkeypatch):
    solver, client = _consensus(
        tmp_path, monkeypatch, 2,
        A=[_writer(1.5), "VERDICT: PASS"], B=[_writer(2.5)], judge=["B is right.\nCHOICE: B"],
    )  # fmt: skip
    assert solver.run()
    assert _out(tmp_path) == '{"price": 2.5}'
    assert client.request_count == 4  # A, A's review, B, judge


def test_parallel_consensus_medoid_picks_the_most_supported(tmp_path, monkeypatch):
    def writer(p, q):
        return WRITE_OK.replace('{"price": 1.5}', f'{{"p": {p}, "q": {q}}}')

    solver, _ = _consensus(
        tmp_path, monkeypatch, 3,
        A=[writer(1, 1), "VERDICT: PASS"], B=[writer(1, 2)], C=[writer(3, 2)],
    )  # fmt: skip
    assert solver.run()
    assert _out(tmp_path) == '{"p": 1, "q": 2}'  # B agrees partly with both others


def test_parallel_consensus_candidates_cannot_clobber_each_other(tmp_path, monkeypatch):
    crash = WRITE_OK.replace("'{\"price\": 1.5}')", "'{\"price\": 9}'); raise SystemExit(1)")
    solver, _ = _consensus(
        tmp_path, monkeypatch, 2, A=[_writer(1.5), "VERDICT: PASS"], B=[crash] * 8,
    )  # fmt: skip
    assert solver.run()
    assert _out(tmp_path) == '{"price": 1.5}'  # B failed; its writes stayed in its staging dir


def test_skills_helpers(tmp_path):
    import json
    import math

    import numpy as np
    import pandas as pd

    from agent import skills

    skills.write_json(
        tmp_path / "r.json",
        {
            "a": np.float64(1.5),
            "b": float("nan"),
            "c": np.array([1, 2]),
            "d": pd.Timestamp("2024-01-31"),
            "e": np.inf,
        },
    )
    assert json.loads((tmp_path / "r.json").read_text()) == {
        "a": 1.5, "b": None, "c": [1, 2], "d": "2024-01-31T00:00:00", "e": None,
    }  # fmt: skip
    returns = [0.10, -0.20, 0.05]
    assert math.isclose(skills.max_drawdown(returns), 0.88 / 1.1 - 1.0)  # peak 1.1 -> trough 0.88
    assert math.isclose(skills.max_drawdown([100, 120, 90, 130], is_returns=False), -0.25)
    assert math.isclose(skills.annualized_return([0.01] * 12, 12), 1.01**12 - 1)
    r = pd.Series([0.01, 0.02, -0.01, 0.03])
    assert math.isclose(skills.annualized_vol(r, 12), r.std(ddof=1) * math.sqrt(12))
    assert math.isclose(skills.sharpe_ratio(r, 12), r.mean() / r.std(ddof=1) * math.sqrt(12))
    assert skills.cumulative_returns([0.1, 0.1]).round(10).tolist() == [0.1, 0.21]
    (tmp_path / "t.tsv").write_text("date\tv\n2024-02-01\t2\n2024-01-01\t1\n")
    indexed = skills.to_datetime_index(skills.read_table(tmp_path / "t.tsv"), "date")
    assert indexed["v"].tolist() == [1, 2]


def test_hybrid_reasoning_per_phase(monkeypatch):
    from agent.client import HouseModelClient

    monkeypatch.setenv("HOUSE_REASONING", "hybrid")
    client = HouseModelClient(base_url="http://127.0.0.1:9", model_name="x")
    assert client._mode_for("generate") == "low"
    assert client._mode_for("review") == "off" and client._mode_for("repair") == "off"
    client._check_low_effort("thinking " * 1000 + "</think>```python\nx\n```", "stop")
    assert client._mode_for("generate") == "off"  # not honoured -> generation falls back
    assert client.reasoning_mode == "hybrid"


FAILING_VERIFIER = "```python\nprint('FAIL: price expected 2.5 got 1.5')\nraise SystemExit(1)\n```"
PASSING_VERIFIER = "```python\nprint('PASS: price')\n```"


def test_verifier_failure_drives_a_repair(tmp_path, monkeypatch):
    from agent import loop

    monkeypatch.setattr(loop, "VERIFY", True)
    fixed = _writer(2.5)
    client = ScriptedClient(WRITE_OK, FAILING_VERIFIER, fixed, PASSING_VERIFIER, "VERDICT: PASS")
    assert _solver(tmp_path, client).run()
    assert client.request_count == 5  # script, verifier, repair, verifier, review
    assert "FAIL: price expected 2.5 got 1.5" in client.prompts[2]
    assert _out(tmp_path) == '{"price": 2.5}'


def test_model_may_keep_its_script_against_a_wrong_verifier(tmp_path, monkeypatch):
    from agent import loop

    monkeypatch.setattr(loop, "VERIFY", True)
    client = ScriptedClient(WRITE_OK, FAILING_VERIFIER, WRITE_OK)  # returns the same script
    assert _solver(tmp_path, client).run()
    assert client.request_count == 3
    assert _out(tmp_path) == '{"price": 1.5}'


def test_requirements_checklist_is_requested_once_and_used(tmp_path, monkeypatch):
    from agent import loop

    monkeypatch.setattr(loop, "PLAN", True)
    client = ScriptedClient("1. r.json must contain `price`", WRITE_OK, "VERDICT: PASS")
    assert _solver(tmp_path, client).run()
    assert client.request_count == 3
    assert (
        "REQUIREMENTS CHECKLIST" in client.prompts[1]
        and "must contain `price`" in client.prompts[1]
    )


def test_exploration_output_reaches_the_next_prompt(tmp_path, monkeypatch):
    from agent import loop

    monkeypatch.setattr(loop, "EXPLORE_STEPS", 2)
    client = ScriptedClient("```explore\nprint(6 * 7)\n```", WRITE_OK, "VERDICT: PASS")
    assert _solver(tmp_path, client).run()
    assert client.request_count == 3
    assert "EXPLORATION SO FAR" in client.prompts[1] and "42" in client.prompts[1]
    assert _out(tmp_path) == '{"price": 1.5}'
    assert not (tmp_path / "out" / "42").exists()
