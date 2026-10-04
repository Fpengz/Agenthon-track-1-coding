"""Generic output tests for frequent failure kinds: detection, repair feedback, consensus."""

import json

import pytest
from tests.test_agent_evidence import RecordedClient, default_policy  # noqa: F401 (autouse)
from tests.test_base_agent import RoutedClient

from agent import loop
from agent.loop import AgentSolver
from agent.output_tests import output_test_failures


def failures(tmp_path, files):
    for name, content in files.items():
        text = json.dumps(content) if name.endswith(".json") else content
        (tmp_path / name).write_text(text)
    return "\n".join(output_test_failures(tmp_path))


@pytest.mark.parametrize(
    "files, flagged",
    [
        ({"s.json": {"max_residual_bps": 8948.4}}, "self-reported errors"),
        ({"s.json": {"max_relative_error": 0.79}}, "self-reported errors"),
        ({"s.json": {"american_put": 0.08, "european_put": 10.2}}, "American value below"),
        (
            {"s.json": {"swaptions": [{"name": f"s{i}", "price_bps": 43.04} for i in range(5)]}},
            "same for every instrument",
        ),
        ({"s.json": {"zero_rate_5y": 0.04, "zero_rate_10y": 0.041, "par_rate_10y": 2.09}}, "percent"),
        (
            {"p.csv": "strike,call_price\n90,12.0\n100,13.5\n110,4.0\n120,2.0\n"},
            "not monotone in the strike",
        ),
        (
            {"transition_matrix.csv": "from,A,B,D\nA,0.9,0.05,0.01\nB,0.1,0.8,0.1\n"},
            "do not sum to one",
        ),
        ({"s.json": {"mc_validates": False}}, "validation check as FAILED"),
        # Correct-looking outputs pass every test.
        (
            {
                "s.json": {
                    "max_residual_bps": 1e-9,
                    "max_mc_var_error_pct": 5.2,  # a percent, under 50
                    "american_put": 10.5,
                    "european_put": 10.2,
                    "zero_rate_5y": 0.04,
                    "zero_rate_10y": 0.041,
                    "log_rate_mean": -3.5,
                    "trades": [{"entry_price": 68.2, "pnl": i + 0.5} for i in range(6)],
                },
                "p.csv": "maturity,strike,call_price,put_price\n"
                + "".join(
                    f"{t},{k},{max(1.0, 120 - k) * t:.3f},{max(1.0, k - 80) * t:.3f}\n"
                    for t in (0.5, 1.0)
                    for k in (90, 100, 110)
                ),
                "transition_matrix.csv": "from,A,B,D\nA,0.9,0.09,0.01\nB,0.1,0.8,0.1\n",
            },
            None,
        ),
    ],
)
def test_output_tests(tmp_path, files, flagged):
    found = failures(tmp_path, files)
    assert flagged in found if flagged else not found


def test_put_call_column_layout_is_checked_per_side(tmp_path):
    rows = "type,strike,price\n" + "".join(
        f"{s},{k},{p}\n" for s, k, p in [("C", 90, 12), ("C", 100, 7), ("C", 110, 9),
                                        ("P", 90, 2), ("P", 100, 5), ("P", 110, 9)]
    )  # fmt: skip
    found = failures(tmp_path, {"opts.csv": rows})
    assert "call price rises with strike at 1 of 2" in found


def output(data):
    return (
        "```python\nimport os, pathlib\n"
        "(pathlib.Path(os.environ['OUTPUT_DIR']) / 'r.json').write_text("
        + repr(json.dumps(data))
        + ")\n```"
    )


def solver_for(tmp_path, client):
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("Price the options; write /output/r.json.")
    return AgentSolver(task, tmp_path / "out", client=client)


def test_failed_test_drives_a_root_cause_repair(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "OUTPUT_TESTS", True)
    monkeypatch.setattr(loop, "CANDIDATES", 1)
    client = RecordedClient(
        output({"american_put": 0.08, "european_put": 10.2}),
        output({"american_put": 10.5, "european_put": 10.2}),
        "VERDICT: PASS",
    )
    solver = solver_for(tmp_path, client)
    assert solver.run()
    assert json.loads((solver.out_dir / "r.json").read_text())["american_put"] == 10.5
    assert [kwargs["phase"] for _, kwargs in client.requests] == ["generate", "repair", "review"]
    repair = client.requests[1][0]
    assert "TEST FAILED: American value below European value" in repair
    assert "ROOT CAUSE" in repair and "american_put = 0.08" in repair


def test_output_tests_off_by_default(tmp_path, monkeypatch):
    assert loop.OUTPUT_TESTS is False
    monkeypatch.setattr(loop, "CANDIDATES", 1)
    client = RecordedClient(output({"american_put": 0.08, "european_put": 10.2}), "VERDICT: PASS")
    assert solver_for(tmp_path, client).run()
    assert [kwargs["phase"] for _, kwargs in client.requests] == ["generate", "review"]


def test_consensus_never_lets_failing_candidates_agree(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "OUTPUT_TESTS", True)
    monkeypatch.setattr(loop, "CANDIDATES", 3)
    monkeypatch.setattr(loop, "MAX_GUARD_REPAIRS", 0)
    bad = output({"american_put": 0.08, "european_put": 10.2})
    good = output({"american_put": 10.5, "european_put": 10.2})
    client = RoutedClient(A=[bad, "VERDICT: PASS"], B=[bad], C=[good])
    solver = solver_for(tmp_path, client)
    assert solver.run()
    # A and B agree exactly, but both fail a test; C is the only candidate that passes.
    assert json.loads((solver.out_dir / "r.json").read_text())["american_put"] == 10.5


def test_consensus_falls_back_when_every_candidate_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "OUTPUT_TESTS", True)
    monkeypatch.setattr(loop, "CANDIDATES", 2)
    monkeypatch.setattr(loop, "MAX_GUARD_REPAIRS", 0)
    bad = output({"american_put": 0.08, "european_put": 10.2})
    client = RoutedClient(A=[bad], B=[bad], judge=["CHOICE: A"])
    solver = solver_for(tmp_path, client)
    assert solver.run()
    assert json.loads((solver.out_dir / "r.json").read_text())["american_put"] == 0.08


# ---- property tests written by the agent ----

PROPERTY_SUITE = (
    "```python\nimport json, os, pathlib\n"
    "p = json.loads((pathlib.Path(os.environ['OUTPUT_DIR']) / 'r.json').read_text())['price']\n"
    "ok = True\n"
    "for floor in (1.8, 2.0):\n"
    "    if p > floor: print(f'PASS: price above {floor}')\n"
    "    else: print(f'FAIL: price {p} not above {floor}'); ok = False\n"
    "raise SystemExit(0 if ok else 1)\n```"
)


def price_writer(value):
    return output({"price": value})


class PropertyClient(RoutedClient):
    """Consensus client that answers the test-writing request with a fixed suite."""

    def __init__(self, suite, **per_candidate):
        super().__init__(**per_candidate)
        self.suite = suite
        self.phases = []

    def chat(self, messages, **kwargs):
        self.phases.append(kwargs.get("phase"))
        if kwargs.get("phase") == "tests":
            with self.lock:
                self.request_count += 1
                self.test_prompt = messages[-1]["content"]
            from agent.client import ChatResult

            return ChatResult(self.suite, "stop")
        return super().chat(messages, **kwargs)


def test_property_tests_drive_a_repair(tmp_path, monkeypatch):
    from tests.test_agent_evidence import SpecTestClient

    monkeypatch.setattr(loop, "PROPERTY_TESTS", True)
    client = SpecTestClient(PROPERTY_SUITE, price_writer(1.5), price_writer(2.5), "VERDICT: PASS")
    solver = solver_for(tmp_path, client)
    assert solver.run()
    assert json.loads((solver.out_dir / "r.json").read_text())["price"] == 2.5
    test_prompt = next(p for p, k in client.requests if k["phase"] == "tests")
    assert "PROPERTY TESTS" in test_prompt and "American >= European" in test_prompt
    repair = next(p for p, k in client.requests if k["phase"] == "repair")
    assert "FAIL: price 1.5 not above 1.8" in repair


def test_consensus_candidates_failing_property_tests_cannot_agree(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "PROPERTY_TESTS", True)
    monkeypatch.setattr(loop, "CANDIDATES", 3)
    monkeypatch.setattr(loop, "MAX_TEST_REPAIRS", 0)
    client = PropertyClient(
        PROPERTY_SUITE,
        A=[price_writer(1.5), "VERDICT: PASS"],
        B=[price_writer(1.5)],
        C=[price_writer(2.5)],
    )
    solver = solver_for(tmp_path, client)
    assert solver.run()
    assert json.loads((solver.out_dir / "r.json").read_text())["price"] == 2.5
    assert client.phases.count("tests") == 1  # one suite shared by every candidate
    assert "PROPERTY TESTS" in client.test_prompt


def test_when_every_candidate_fails_the_fewest_failures_win(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "PROPERTY_TESTS", True)
    monkeypatch.setattr(loop, "CANDIDATES", 2)
    monkeypatch.setattr(loop, "MAX_TEST_REPAIRS", 0)
    client = PropertyClient(
        PROPERTY_SUITE, A=[price_writer(1.5), "VERDICT: PASS"], B=[price_writer(1.9)]
    )
    solver = solver_for(tmp_path, client)
    assert solver.run()
    assert json.loads((solver.out_dir / "r.json").read_text())["price"] == 1.9  # fails 1 of 2
