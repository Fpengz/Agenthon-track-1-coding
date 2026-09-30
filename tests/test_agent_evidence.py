"""Regression tests for the agent's review, verification and output fallback paths."""

import json
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
    for name in ("VERIFY", "PLAN", "SKILLS", "ADAPTIVE", "SPEC_CHECKS", "EXAMPLES"):
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
