"""Tests for the base agent implementation."""

import pathlib
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


def test_domain_notes_pick_the_matching_category(tmp_path):
    from agent.knowledge import domain_notes

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
    monkeypatch.setenv("AGENT_DOMAIN_NOTES", "0")
    assert domain_notes(tmp_path, "Bootstrap a zero coupon yield curve.") == ""
    monkeypatch.setenv("AGENT_DOMAIN_NOTES", "1")
    assert "fixed-income" in domain_notes(tmp_path, "Bootstrap a zero coupon yield curve.")
