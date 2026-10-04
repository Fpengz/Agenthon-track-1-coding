"""Regression tests for the agent's review, verification and output fallback paths."""

import json
import threading
import time

import pytest

from agent import loop
from agent.client import ChatResult
from agent.executor import ExecutionResult, extract_python_code
from agent.loop import AgentSolver
from agent.review import parse_verdict, verifier_result


def writer(value):
    return (
        "```python\nimport os, pathlib\n"
        "out = pathlib.Path(os.environ['OUTPUT_DIR'])\n"
        f"(out / 'r.json').write_text('{{\"price\": {value}}}')\n```"
    )


class RecordedClient:
    max_requests = 25

    def __init__(self, *responses):
        self.responses = list(responses)
        self.request_count = 0
        self.requests = []
        self.systems = []
        self.after_response = None

    def chat(self, messages, **kwargs):
        self.request_count += 1
        self.requests.append((messages[-1]["content"], kwargs))
        self.systems.append(messages[0]["content"])
        response = self.responses.pop(0)
        if self.after_response:
            self.after_response(self.request_count)
        return response if isinstance(response, ChatResult) else ChatResult(response, "stop")


@pytest.fixture(autouse=True)
def default_policy(monkeypatch):
    for name in (
        "VERIFY",
        "TESTS",
        "PLAN",
        "SKILLS",
        "ADAPTIVE",
        "SPEC_CHECKS",
        "EXAMPLES",
        "GUARDRAILS",
        "REPAIR_V2",
        "OUTPUT_TESTS",
        "PROPERTY_TESTS",
    ):
        monkeypatch.setattr(loop, name, False, raising=False)
    monkeypatch.setattr(loop, "CANDIDATES", 1)
    monkeypatch.setattr(loop, "EXPLORE_STEPS", 0)


def solver_for(tmp_path, client):
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("Write /output/r.json with price equal to 2.5.")
    return AgentSolver(task, tmp_path / "out", client=client)


def price(tmp_path):
    return json.loads((tmp_path / "out" / "r.json").read_text())["price"]


@pytest.mark.parametrize("finish_reason", ["length", "stop"])
@pytest.mark.parametrize("marker_width", [5, 7, 9])
def test_repair_v2_never_executes_a_truncated_edit_batch(
    tmp_path, monkeypatch, finish_reason, marker_width
):
    monkeypatch.setattr(loop, "REPAIR_V2", True, raising=False)
    bad = "```python\n" + "# line\n" * 80 + "raise ValueError('original_error')\n```"
    partial = (
        "<<<<<<< SEARCH\nraise ValueError('original_error')\n=======\n"
        + extract_python_code(writer(1.5))
        + "\n>>>>>>> REPLACE\n<<<<<<< SEARCH\nunfinished"
    )
    partial = (
        partial.replace("<<<<<<<", "<" * marker_width)
        .replace("=======", "=" * marker_width)
        .replace(">>>>>>>", ">" * marker_width)
    )
    client = RecordedClient(bad, ChatResult(partial, finish_reason), writer(2.5))
    solver = solver_for(tmp_path, client)
    solver.max_retries = 2
    assert solver.run()
    assert client.requests[2][1]["phase"] == "repair"
    assert "truncated" in client.requests[2][0].lower()
    assert price(tmp_path) == 2.5


def test_repair_v2_preserves_static_failure_when_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "REPAIR_V2", True, raising=False)
    bad = "```python\nimport unavailable_example_package\n```"
    client = RecordedClient(bad, bad, writer(2.5))
    solver = solver_for(tmp_path, client)
    solver.max_retries = 2
    assert solver.run()
    prompt = client.requests[2][0]
    assert "unavailable_example_package" in prompt
    assert "not installed" in prompt


def test_repair_v2_exhausted_continuations_switch_to_compact_code(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "REPAIR_V2", True)
    client = RecordedClient(
        ChatResult("```python\nimport os\n", "length"),
        ChatResult("```python\nx = 1\n", "length"),
        ChatResult("```python\ny = 2\n", "length"),
        writer(2.5),
    )
    solver = solver_for(tmp_path, client)
    solver.max_retries = 3
    assert solver.run()
    assert [kw["phase"] for _, kw in client.requests] == [
        "generate",
        "continue",
        "continue",
        "compact",
    ]
    assert "EVERY required deliverable" in client.requests[-1][0]
    assert price(tmp_path) == 2.5


def test_repair_v2_closed_prefix_of_truncated_solution_is_not_run(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "REPAIR_V2", True)
    client = RecordedClient(
        ChatResult(writer(1.5) + "\n```python\nunfinished", "length"), writer(2.5)
    )
    solver = solver_for(tmp_path, client)
    solver.max_retries = 1

    def before_next_request(number):
        if number == 2:
            assert not (solver.out_dir / "r.json").exists()

    client.after_response = before_next_request
    assert solver.run()
    assert client.requests[1][1]["phase"] == "compact"


def test_repair_v2_repeated_execution_error_changes_repair_strategy(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "REPAIR_V2", True)
    bad = "```python\nraise ValueError('same_failure')\n```"
    different_but_bad = "```python\nx = 1\nraise ValueError('same_failure')\n```"
    client = RecordedClient(bad, different_but_bad, writer(2.5))
    solver = solver_for(tmp_path, client)
    solver.max_retries = 2
    assert solver.run()
    assert client.requests[2][1]["phase"] == "compact"
    assert "same_failure" in client.requests[2][0]


def test_repair_v2_numeric_audit_repairs_and_rechecks_frozen_test(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "REPAIR_V2", True, raising=False)
    client = RecordedClient(writer(1.5), PRICE_TEST, writer(1.75), writer(2.5))
    solver = solver_for(tmp_path, client)
    solver.max_retries = 2
    assert solver.run()
    assert price(tmp_path) == 2.5
    assert client.requests[1][1]["phase"] == "audit"
    assert "SCRIPT THAT PRODUCED" not in client.requests[1][0]
    assert "price expected 2.5 got 1.75" in client.requests[3][0]


def test_repair_v2_input_only_audit_has_no_output_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "REPAIR_V2", True)
    client = RecordedClient(
        writer(2.5),
        "```python\nassert False, 'input_only'\n```",
        "```python\nprint('PASS: inputs')\n```",
        "VERDICT: PASS",
    )
    solver = solver_for(tmp_path, client)
    assert solver.run()
    assert solver._last_numeric_audit == ""
    assert price(tmp_path) == 2.5


def test_repair_v2_truncated_audit_is_not_executed(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "REPAIR_V2", True)
    partial = PRICE_TEST.replace("value == 2.5", "value == 99")
    client = RecordedClient(writer(2.5), ChatResult(partial, "length"), PRICE_TEST, "VERDICT: PASS")
    solver = solver_for(tmp_path, client)
    assert solver.run()
    assert client.request_count == 4
    assert solver._last_numeric_audit == extract_python_code(PRICE_TEST)


def test_repair_v2_keeps_audited_outputs_after_bad_review_rewrite(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "REPAIR_V2", True)
    client = RecordedClient(writer(2.5), PRICE_TEST, "VERDICT: FAIL\n" + writer(1.5))
    solver = solver_for(tmp_path, client)
    solver.max_retries = 1
    assert solver.run()
    assert price(tmp_path) == 2.5


def test_guardrails_repair_schema_despite_a_model_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "GUARDRAILS", True, raising=False)
    client = RecordedClient(writer(1), "VERDICT: PASS", writer(2.5), "VERDICT: PASS")
    solver = solver_for(tmp_path, client)
    (solver.task_dir / "instruction.md").write_text(
        'Write /output/r.json.\n### r.json\n```json\n{"bond_A": {"units": <float>}}\n```'
    )

    def output(data):
        return "```python\nimport os, pathlib\n" + (
            "(pathlib.Path(os.environ['OUTPUT_DIR']) / 'r.json').write_text("
            + repr(json.dumps(data))
            + ")\n```"
        )

    client.responses[0] = output({"A": {"units": 1}})
    client.responses[2] = output({"bond_A": {"units": 2.5}})
    assert solver.run()
    assert "bond_A" in json.loads((solver.out_dir / "r.json").read_text())
    assert client.requests[1][1]["phase"] == "repair"
    assert "bond_A" in client.requests[1][0]


def test_guardrails_preserve_valid_outputs_after_a_bad_review_fix(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "GUARDRAILS", True, raising=False)
    client = RecordedClient(writer(2.5), "VERDICT: FAIL\n" + writer("NaN"))
    solver = solver_for(tmp_path, client)
    solver.max_retries = 1
    (solver.task_dir / "instruction.md").write_text(
        'Write /output/r.json.\n### r.json\n```json\n{"price": <float>}\n```'
    )
    assert solver.run()
    assert price(tmp_path) == 2.5


def test_guardrails_recover_crashed_probe_before_resuming_solution(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "GUARDRAILS", True, raising=False)
    monkeypatch.setattr(loop, "EXPLORE_STEPS", 1)
    client = RecordedClient(
        "```python\nraise NameError('measurement_missing_import')\n```",
        "```python\nprint('measured_tail_mean=2.5')\n```",
        writer(2.5),
        "VERDICT: PASS",
    )
    assert solver_for(tmp_path, client).run()
    assert [kwargs["phase"] for _, kwargs in client.requests] == [
        "probe",
        "probe",
        "generate",
        "review",
    ]
    assert "measurement_missing_import" in client.requests[1][0]
    assert "measured_tail_mean=2.5" in client.requests[2][0]


def test_guardrails_repair_audited_probe_failure_and_recheck(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "GUARDRAILS", True)
    monkeypatch.setattr(loop, "EXPLORE_STEPS", 2)
    client = RecordedClient(
        "```python\nprint('toy_price=2.5')\n```",
        writer(1.5),
        PRICE_TEST,
        writer(2.5),
        "VERDICT: PASS",
    )
    solver = solver_for(tmp_path, client)
    assert solver.run()
    assert price(tmp_path) == 2.5
    assert [kwargs["phase"] for _, kwargs in client.requests] == [
        "probe",
        "generate",
        "probe",
        "repair",
        "review",
    ]
    assert "price expected 2.5 got 1.5" in client.requests[3][0]


def test_guardrails_probe_recovery_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "GUARDRAILS", True)
    monkeypatch.setattr(loop, "EXPLORE_STEPS", 1)
    bad = "```python\nraise NameError('broken_probe')\n```"
    client = RecordedClient(bad, bad, writer(2.5), "VERDICT: PASS")
    assert solver_for(tmp_path, client).run()
    assert [k["phase"] for _, k in client.requests] == ["probe", "probe", "generate", "review"]


def test_guardrails_unmatched_edits_switch_to_a_complete_rewrite(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "GUARDRAILS", True)
    bad = "```python\n" + "# line\n" * 80 + "raise ValueError('original_error')\n```"
    edits = "<<<<<<< SEARCH\nnot_in_the_script\n=======\nfixed\n>>>>>>> REPLACE"
    client = RecordedClient(bad, edits, edits, writer(2.5), "VERDICT: PASS")
    assert solver_for(tmp_path, client).run()
    assert price(tmp_path) == 2.5
    prompt = client.requests[3][0]
    assert "Rewrite the complete compact script" in prompt
    assert "original_error" in prompt
    assert "entire corrected Python script" in prompt


def test_guardrails_input_only_assertion_cannot_reject_a_solution(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "GUARDRAILS", True)
    monkeypatch.setattr(loop, "EXPLORE_STEPS", 2)
    client = RecordedClient(
        "```python\nprint('toy_price=2.5')\n```",
        writer(2.5),
        "```python\nassert False, 'input_only_hypothesis'\n```",
        "VERDICT: PASS",
    )
    assert solver_for(tmp_path, client).run()
    assert price(tmp_path) == 2.5
    assert client.requests[-1][1]["phase"] == "review"


def test_guardrails_failed_output_probe_runs_again_after_an_inadequate_fix(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "GUARDRAILS", True)
    monkeypatch.setattr(loop, "EXPLORE_STEPS", 2)
    client = RecordedClient(
        "```python\nprint('toy_price=2.5')\n```",
        writer(1.5),
        PRICE_TEST,
        writer(1.7),
        writer(2.5),
        "VERDICT: PASS",
    )
    assert solver_for(tmp_path, client).run()
    assert price(tmp_path) == 2.5
    assert "price expected 2.5 got 1.7" in client.requests[4][0]
    assert client.requests[4][1]["phase"] == "repair"


def test_initial_python_probe_is_not_a_solution_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "EXPLORE_STEPS", 1)
    probe = writer(999).replace("\n```", "\nprint('measured_price=2.5')\n```")
    client = RecordedClient(probe, writer(2.5), "VERDICT: PASS")
    solver = solver_for(tmp_path, client)

    def check_before_solution(number):
        if number == 2:
            assert not (solver.out_dir / "r.json").exists(), "probe wrote real deliverables"

    client.after_response = check_before_solution
    assert solver.run()
    assert price(tmp_path) == 2.5
    assert [kwargs["phase"] for _, kwargs in client.requests] == ["probe", "generate", "review"]
    assert "measured_price=2.5" in client.requests[1][0]
    assert "missing deliverable" not in client.requests[1][0]
    assert client.systems[0] == loop.PROBE_SYSTEM_PROMPT
    assert client.systems[1] == loop.TOOL_SYSTEM_PROMPT
    assert "CURRENT PHASE: NUMERICAL PROBE BEFORE GENERATE" in client.requests[0][0]


@pytest.mark.parametrize("failed_solution", [False, True])
def test_numeric_probe_evidence_survives_repair_and_review(tmp_path, monkeypatch, failed_solution):
    monkeypatch.setattr(loop, "EXPLORE_STEPS", 1)
    responses = ["```explore\nprint('measured_tail_mean=2.5')\n```"]
    if failed_solution:
        responses.append("```python\nraise ValueError('wrong_tail_convention')\n```")
    responses.extend([writer(2.5), "VERDICT: PASS"])
    client = RecordedClient(*responses)
    assert solver_for(tmp_path, client).run()
    followups = [
        prompt for prompt, kwargs in client.requests if kwargs["phase"] in {"repair", "review"}
    ]
    assert followups
    assert all("measured_tail_mean=2.5" in prompt for prompt in followups)


@pytest.mark.parametrize("closing_fence", ["\n```", ""])
@pytest.mark.parametrize("language", ["python", "explore"])
def test_truncated_probe_is_not_executed(tmp_path, monkeypatch, closing_fence, language):
    monkeypatch.setattr(loop, "EXPLORE_STEPS", 1)
    snippet = (
        "import os, pathlib\n(pathlib.Path(os.environ['TASK_DIR']) / 'partial_probe_ran').touch()"
    )
    client = RecordedClient(
        ChatResult(f"```{language}\n{snippet}\n{closing_fence}", "length"),
        "```python\nprint('complete_measurement=2.5')\n```",
        writer(2.5),
        "VERDICT: PASS",
    )
    solver = solver_for(tmp_path, client)
    assert solver.run()
    assert not (solver.task_dir / "partial_probe_ran").exists()
    assert [kwargs["phase"] for _, kwargs in client.requests] == [
        "probe",
        "probe",
        "generate",
        "review",
    ]
    assert "complete_measurement=2.5" in client.requests[2][0]


def test_repair_can_probe_without_losing_the_failure_or_previous_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "EXPLORE_STEPS", 2)
    client = RecordedClient(
        "```explore\nprint('first_measurement=2.5')\n```",
        "```python\nraise ValueError('wrong_tail_convention')\n```",
        "```explore\nprint('independent_tail_integral=2.5')\n```",
        writer(2.5),
        "VERDICT: PASS",
    )
    assert solver_for(tmp_path, client).run()
    prompt, kwargs = client.requests[3]
    assert kwargs["phase"] == "repair"
    assert "first_measurement=2.5" in prompt
    assert "independent_tail_integral=2.5" in prompt
    assert "wrong_tail_convention" in prompt
    assert "raise ValueError('wrong_tail_convention')" in prompt


@pytest.mark.parametrize("language", ["python", "explore"])
def test_review_probe_reads_a_fresh_output_copy_and_resumes_review(tmp_path, monkeypatch, language):
    monkeypatch.setattr(loop, "EXPLORE_STEPS", 2)
    review_probe = (
        f"```{language}\nimport json, os, pathlib\n"
        "p = pathlib.Path(os.environ['OUTPUT_DIR']) / 'r.json'\n"
        "print('observed_price=', json.loads(p.read_text())['price'])\n"
        "p.write_text('{}')\n```"
    )
    client = RecordedClient(
        "```explore\nprint('toy_price=2.5')\n```", writer(2.5), review_probe, "VERDICT: PASS"
    )
    assert solver_for(tmp_path, client).run()
    prompt, kwargs = client.requests[-1]
    assert kwargs["phase"] == "review"
    assert "observed_price= 2.5" in prompt
    assert "SCRIPT STDOUT" in prompt
    assert price(tmp_path) == 2.5
    assert [kwargs["phase"] for _, kwargs in client.requests] == [
        "probe",
        "generate",
        "probe",
        "review",
    ]
    assert client.systems[2] == loop.PROBE_SYSTEM_PROMPT


def test_review_probe_assertion_evidence_drives_a_numerical_fix(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "EXPLORE_STEPS", 2)
    probe = PRICE_TEST.replace("```python", "```explore")
    client = RecordedClient(
        "```explore\nprint('independent_price=2.5')\n```",
        writer(1.5),
        probe,
        "VERDICT: FAIL\n" + writer(2.5),
        "VERDICT: PASS",
    )
    assert solver_for(tmp_path, client).run()
    assert price(tmp_path) == 2.5
    prompt, kwargs = client.requests[3]
    assert kwargs["phase"] == "review"
    assert "AssertionError: price expected 2.5 got 1.5" in prompt
    assert "[exit 1]" in prompt
    assert "output copy: clean solution turn 2" in prompt


def test_probe_cannot_become_a_solution_continuation(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "EXPLORE_STEPS", 2)
    partial = "```python\nimport os, pathlib\nout = pathlib.Path(os.environ['OUTPUT_DIR'])\n"
    client = RecordedClient(
        "```explore\nprint('independent_price=2.5')\n```",
        ChatResult(partial, "length"),
        "```explore\nimport os, pathlib\n(pathlib.Path(os.environ['TASK_DIR']) / 'unexpected_probe').touch()\n```",
        "```python\n(out / 'r.json').write_text('{\"price\": 2.5}')\n```",
        "```python\nprint('review_measurement=2.5')\n```",
        "VERDICT: PASS",
    )
    solver = solver_for(tmp_path, client)
    assert solver.run()
    assert price(tmp_path) == 2.5
    assert not (solver.task_dir / "unexpected_probe").exists()
    assert client.requests[3][1]["phase"] == "continue"


@pytest.mark.parametrize("review_probe", [False, True])
def test_incomplete_explicit_probe_recovers_once_then_resumes(tmp_path, monkeypatch, review_probe):
    monkeypatch.setattr(loop, "EXPLORE_STEPS", 2 if review_probe else 1)
    responses = (
        ["```python\nprint('initial_measurement=2.5')\n```", writer(2.5)] if review_probe else []
    )
    responses.extend(
        [ChatResult("```python\nprint('unfinished')\n```", "length"), "unfinished probe"]
    )
    responses.extend(["VERDICT: PASS"] if review_probe else [writer(2.5), "VERDICT: PASS"])
    client = RecordedClient(*responses)
    assert solver_for(tmp_path, client).run()
    assert price(tmp_path) == 2.5
    phases = [kwargs["phase"] for _, kwargs in client.requests]
    assert phases == (
        ["probe", "generate", "probe", "probe", "review"]
        if review_probe
        else ["probe", "probe", "generate", "review"]
    )
    assert "recovery budget is exhausted" in client.requests[-1 if review_probe else -2][0]
    assert "[stdout]\nunfinished" not in client.requests[-1][0]


def test_expired_explicit_review_probe_preserves_clean_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "EXPLORE_STEPS", 2)
    client = RecordedClient(
        "```python\nprint('initial_measurement=2.5')\n```", writer(2.5), writer(999)
    )
    solver = solver_for(tmp_path, client)

    def expire_after_probe_response(number):
        if number == 3:
            solver.deadline = time.monotonic() - 1

    client.after_response = expire_after_probe_response
    assert solver.run()
    assert price(tmp_path) == 2.5
    assert [kwargs["phase"] for _, kwargs in client.requests] == ["probe", "generate", "probe"]


def test_probe_evidence_survives_context_window_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "EXPLORE_STEPS", 1)
    client = RecordedClient(
        "```explore\nprint('measured_price=2.5')\n```", writer(2.5), "VERDICT: PASS"
    )
    solver = solver_for(tmp_path, client)
    original_variants = solver._prompt_variants

    def rich_variants(*args, **kwargs):
        variants = original_variants(*args, **kwargs)
        return [variants[0] + "\nOPTIONAL LARGE PREVIEW\n" + "x" * 16000, *variants]

    monkeypatch.setattr(solver, "_prompt_variants", rich_variants)

    def shrink_after_probe(number):
        if number == 1:
            monkeypatch.setattr(loop, "CONTEXT_TOKENS", 6500)

    client.after_response = shrink_after_probe
    assert solver.run()
    assert "OPTIONAL LARGE PREVIEW" in client.requests[0][0]
    assert "OPTIONAL LARGE PREVIEW" not in client.requests[1][0]
    assert all("measured_price=2.5" in prompt for prompt, _ in client.requests[1:])


def test_exhausted_probe_allowance_keeps_clean_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "EXPLORE_STEPS", 1)
    client = RecordedClient(
        "```explore\nprint('measured_price=2.5')\n```",
        writer(2.5),
        "```explore\nimport os, pathlib\n(pathlib.Path(os.environ['TASK_DIR']) / 'extra_probe_ran').touch()\n```",
        "VERDICT: PASS",
    )
    solver = solver_for(tmp_path, client)
    assert solver.run()
    assert not (solver.task_dir / "extra_probe_ran").exists()
    assert price(tmp_path) == 2.5
    assert "was NOT executed" in client.requests[-1][0]
    assert client.requests[-1][1]["phase"] == "review"


@pytest.mark.parametrize("same_error", [True, False])
def test_repeated_failure_detection_uses_the_exception_not_local_variables(tmp_path, same_error):
    first = "```python\nmapping = {'a': 1}\nvalue = mapping['missing']\n```"
    second = (
        "```python\nmapping = {'b': 1}\nvalue = mapping['missing']\n```"
        if same_error
        else "```python\nmapping = {'a': 1}\nvalue = mapping['other_missing']\n```"
    )
    client = RecordedClient(first, second, writer(2.5), "VERDICT: PASS")
    assert solver_for(tmp_path, client).run()
    repair, kwargs = client.requests[2]
    assert kwargs["phase"] == "repair"
    assert ("SAME error" in repair) == same_error


class SpecTestClient(RecordedClient):
    """Synchronise the two initial calls to prove generation overlaps test writing."""

    def __init__(self, test_response, *responses):
        super().__init__(*responses)
        self.test_response = test_response
        self.tests_started = threading.Event()
        self.solution_started = threading.Event()
        self.lock = threading.Lock()

    def chat(self, messages, **kwargs):
        with self.lock:
            self.request_count += 1
            self.requests.append((messages[-1]["content"], kwargs))
            tests = kwargs["phase"] == "tests"
            response = self.test_response if tests else self.responses.pop(0)
        if tests:
            self.tests_started.set()
            assert self.solution_started.wait(timeout=3), "test writing was sequential"
        elif kwargs["phase"] in {"generate", "probe"}:
            self.solution_started.set()
            assert self.tests_started.wait(timeout=3), "solution writing was sequential"
        return response if isinstance(response, ChatResult) else ChatResult(response, "stop")


PRICE_TEST = (
    "```python\nimport json, os, pathlib\n"
    "value = json.loads((pathlib.Path(os.environ['OUTPUT_DIR']) / 'r.json').read_text())['price']\n"
    "assert value == 2.5, f'price expected 2.5 got {value}'\n"
    "print('PASS: price')\n```"
)


def test_spec_first_tests_overlap_generation_and_recheck_a_frozen_suite(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "TESTS", True)
    client = SpecTestClient(PRICE_TEST, writer(1.5), writer(2.5), "VERDICT: PASS")
    assert solver_for(tmp_path, client).run()
    assert price(tmp_path) == 2.5
    assert client.request_count == 4
    test_requests = [prompt for prompt, kwargs in client.requests if kwargs["phase"] == "tests"]
    assert len(test_requests) == 1
    assert "TEST AUTHOR" in test_requests[0]
    assert "1.5" not in test_requests[0]
    assert "SCRIPT THAT PRODUCED" not in test_requests[0]
    repair = next(prompt for prompt, kwargs in client.requests if kwargs["phase"] == "repair")
    assert "price expected 2.5 got 1.5" in repair
    assert all("deadline" in kwargs for _, kwargs in client.requests)


def test_spec_first_tests_can_overlap_an_explicit_initial_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "TESTS", True)
    monkeypatch.setattr(loop, "EXPLORE_STEPS", 1)
    client = SpecTestClient(
        PRICE_TEST,
        "```python\nprint('independent_price=2.5')\n```",
        writer(1.5),
        writer(2.5),
        "VERDICT: PASS",
    )
    assert solver_for(tmp_path, client).run()
    assert price(tmp_path) == 2.5
    assert client.request_count == 5
    assert sum(kwargs["phase"] == "probe" for _, kwargs in client.requests) == 1
    test_prompt = next(prompt for prompt, kwargs in client.requests if kwargs["phase"] == "tests")
    assert "independent_price" not in test_prompt
    assert "EXPLORATION SO FAR" not in test_prompt


@pytest.mark.parametrize(
    "suite",
    [
        "```python\nraise KeyError('test_bug')\n```",
        "```python\npass\n```",
        ChatResult("```python\nassert False\n```", "length"),
        "Incomplete test reasoning, no script",
    ],
)
def test_incomplete_spec_first_tests_fall_back_to_review(tmp_path, monkeypatch, suite):
    monkeypatch.setattr(loop, "TESTS", True)
    client = SpecTestClient(suite, writer(2.5), "VERDICT: PASS")
    assert solver_for(tmp_path, client).run()
    assert price(tmp_path) == 2.5
    assert client.request_count == 3
    assert not any(kwargs["phase"] == "repair" for _, kwargs in client.requests)


def test_spec_first_tests_cannot_corrupt_the_deliverables(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "TESTS", True)
    modifies_copy = (
        "```python\nimport os, pathlib\n"
        "(pathlib.Path(os.environ['OUTPUT_DIR']) / 'r.json').write_text('{}')\n"
        "raise KeyError('test_bug')\n```"
    )
    client = SpecTestClient(modifies_copy, writer(2.5), "VERDICT: PASS")
    assert solver_for(tmp_path, client).run()
    assert price(tmp_path) == 2.5


def test_solution_can_reject_an_incorrect_spec_first_test(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "TESTS", True)
    bad_test = (
        "```python\nimport os, pathlib\n"
        "(pathlib.Path(os.environ['OUTPUT_DIR']) / 'r.json').read_text()\n"
        "assert False, 'bad assumption'\n```"
    )
    client = SpecTestClient(bad_test, writer(2.5), writer(2.5))
    assert solver_for(tmp_path, client).run()
    assert price(tmp_path) == 2.5
    assert client.request_count == 3


@pytest.mark.parametrize("reported", ["print('PASS: inputs')", "assert False, 'wrong input'"])
def test_input_only_suites_cannot_produce_output_evidence(tmp_path, monkeypatch, reported):
    monkeypatch.setattr(loop, "TESTS", True)
    suite = f"```python\nimport os\nunused = os.environ['OUTPUT_DIR']\n{reported}\n```"
    client = SpecTestClient(suite, writer(2.5), "VERDICT: PASS")
    assert solver_for(tmp_path, client).run()
    assert price(tmp_path) == 2.5
    assert client.request_count == 3
    assert not any(kwargs["phase"] == "repair" for _, kwargs in client.requests)


def test_spec_first_suite_preserves_future_imports_and_reads_binary_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "TESTS", True)
    suite = (
        "```python\nfrom __future__ import annotations\nimport os, pathlib\n"
        "actual = (pathlib.Path(os.environ['OUTPUT_DIR']) / 'r.json').read_bytes()\n"
        "assert b'2.5' in actual, 'wrong price'\nprint('PASS: price')\n```"
    )
    client = SpecTestClient(suite, writer(1.5), writer(2.5), "VERDICT: PASS")
    assert solver_for(tmp_path, client).run()
    assert price(tmp_path) == 2.5
    assert client.request_count == 4


def test_spec_first_test_repairs_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "TESTS", True)
    client = SpecTestClient(PRICE_TEST, writer(1.5), writer(1.75), writer(2.0), "VERDICT: PASS")
    assert solver_for(tmp_path, client).run()
    assert price(tmp_path) == 2.0
    assert sum(kwargs["phase"] == "tests" for _, kwargs in client.requests) == 1
    assert sum(kwargs["phase"] == "repair" for _, kwargs in client.requests) == 2


def test_spec_first_test_setting_is_recorded_and_forwarded():
    from agent.batch import AGENT_ENV_VARS
    from agent.experiments import SETTING_VARS

    assert "AGENT_TESTS" in AGENT_ENV_VARS
    assert "AGENT_TESTS" in SETTING_VARS


def test_reasoning_draft_is_not_a_final_script():
    response = writer(1.5) + "\nThis is only a draft.\n</think>\nThe final script needs changes."
    assert extract_python_code(response) == ""


def test_verdict_must_come_from_the_final_answer():
    assert parse_verdict("I might say VERDICT: PASS.\n</think>\nThe review is unfinished.") is None


def test_empty_pass_line_cannot_swallow_a_following_failure(tmp_path):
    run = ExecutionResult(0, "PASS:\nFAIL: price expected 2.5 got 1.5\n", "", tmp_path / "v.py")
    assert verifier_result(run).failures == ["FAIL: price expected 2.5 got 1.5"]


@pytest.mark.parametrize(
    "incomplete",
    [
        ChatResult("The price calculation needs closer inspection", "length"),
        ChatResult("</think>\nThe review is unfinished.", "stop"),
        ChatResult("</think>\nVERDICT: PASS\nStill checking", "length"),
    ],
)
def test_incomplete_review_requests_a_verdict_before_acceptance(tmp_path, incomplete):
    client = RecordedClient(
        writer(1.5), incomplete, "VERDICT: FAIL\n" + writer(2.5), "VERDICT: PASS"
    )
    assert solver_for(tmp_path, client).run()
    assert price(tmp_path) == 2.5
    assert client.request_count == 4
    assert client.requests[2][1]["phase"] == "review"


def test_incomplete_review_recovery_is_bounded_and_preserves_outputs(tmp_path, caplog):
    truncated = ChatResult("Still checking the calculation", "length")
    client = RecordedClient(writer(1.5), truncated, truncated)
    assert solver_for(tmp_path, client).run()
    assert price(tmp_path) == 1.5
    assert client.request_count == 3
    assert "review passed" not in caplog.text


def test_verifier_assertion_is_executed_failure_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "VERIFY", True)
    assertion = "```python\nassert 1.5 == 2.5, 'price expected 2.5 got 1.5'\n```"
    client = RecordedClient(
        writer(1.5), assertion, writer(2.5), "```python\nprint('PASS: price')\n```", "VERDICT: PASS"
    )
    assert solver_for(tmp_path, client).run()
    assert price(tmp_path) == 2.5
    assert "price expected 2.5 got 1.5" in client.requests[2][0]


def test_verifier_crash_repairs_the_verifier_on_a_fresh_output_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "VERIFY", True)
    crashed = (
        "```python\nimport os, pathlib\n"
        "(pathlib.Path(os.environ['OUTPUT_DIR']) / 'r.json').write_text('{}')\n"
        "raise KeyError('wrong_verifier_key')\n```"
    )
    repaired = (
        "```python\nimport os, pathlib, json\n"
        "data = json.loads((pathlib.Path(os.environ['OUTPUT_DIR']) / 'r.json').read_text())\n"
        "print('FAIL: price expected 2.5 got', data['price'])\n```"
    )
    client = RecordedClient(
        writer(1.5),
        crashed,
        repaired,
        writer(2.5),
        "```python\nprint('PASS: price')\n```",
        "VERDICT: PASS",
    )
    assert solver_for(tmp_path, client).run()
    assert price(tmp_path) == 2.5
    assert client.requests[2][1]["phase"] == "verify"
    assert "wrong_verifier_key" in client.requests[2][0]
    assert "got 1.5" in client.requests[3][0]


def test_silent_verifier_is_retried_once_then_reviewed(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(loop, "VERIFY", True)
    client = RecordedClient(
        writer(1.5), "```python\npass\n```", "```python\npass\n```", "VERDICT: PASS"
    )
    assert solver_for(tmp_path, client).run()
    assert client.request_count == 4
    assert price(tmp_path) == 1.5
    assert "incomplete" in caplog.text.lower()


def test_truncated_verifier_cannot_pass_on_only_its_first_block(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "VERIFY", True)
    truncated = ChatResult("```python\nprint('PASS: schema')\n```\n```python\nassert ", "length")
    client = RecordedClient(
        writer(1.5),
        truncated,
        "```python\nassert False, 'wrong price'\n```",
        writer(2.5),
        "```python\nprint('PASS: price')\n```",
        "VERDICT: PASS",
    )
    assert solver_for(tmp_path, client).run()
    assert price(tmp_path) == 2.5
    assert client.requests[2][1]["phase"] == "verify"
    assert "truncated" in client.requests[2][0]
    assert "wrong price" in client.requests[3][0]


def test_verifier_recovery_reserves_a_request_for_solution_repair(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "VERIFY", True)
    client = RecordedClient(writer(1.5), "```python\nraise KeyError('verifier_bug')\n```")
    client.max_requests = 4

    def count_a_transient_retry(number):
        if number == 2:
            client.request_count += 1

    client.after_response = count_a_transient_retry
    assert solver_for(tmp_path, client).run()
    assert client.request_count == 3
    assert len(client.requests) == 2
    assert price(tmp_path) == 1.5


def test_expired_budget_does_not_delete_or_execute_over_clean_outputs(tmp_path):
    client = RecordedClient(writer(1.5), "VERDICT: FAIL\n" + writer(2.5))
    solver = solver_for(tmp_path, client)

    def expire_after_review(number):
        if number == 2:
            solver.deadline = time.monotonic() - 1

    client.after_response = expire_after_review
    assert solver.run()
    assert price(tmp_path) == 1.5
    assert client.request_count == 2
    assert all("deadline" in kwargs for _, kwargs in client.requests)
