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
        self.after_response = None

    def chat(self, messages, **kwargs):
        self.request_count += 1
        self.requests.append((messages[-1]["content"], kwargs))
        response = self.responses.pop(0)
        if self.after_response:
            self.after_response(self.request_count)
        return response if isinstance(response, ChatResult) else ChatResult(response, "stop")


@pytest.fixture(autouse=True)
def default_policy(monkeypatch):
    for name in ("VERIFY", "TESTS", "PLAN", "SKILLS", "ADAPTIVE", "SPEC_CHECKS", "EXAMPLES"):
        monkeypatch.setattr(loop, name, False)
    monkeypatch.setattr(loop, "CANDIDATES", 1)
    monkeypatch.setattr(loop, "EXPLORE_STEPS", 0)


def solver_for(tmp_path, client):
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("Write /output/r.json with price equal to 2.5.")
    return AgentSolver(task, tmp_path / "out", client=client)


def price(tmp_path):
    return json.loads((tmp_path / "out" / "r.json").read_text())["price"]


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
        elif kwargs["phase"] == "generate":
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
    client = SpecTestClient(
        "```python\nassert False, 'bad assumption'\n```", writer(2.5), writer(2.5)
    )
    assert solver_for(tmp_path, client).run()
    assert price(tmp_path) == 2.5
    assert client.request_count == 3


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
