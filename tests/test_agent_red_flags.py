"""Red flags visible in the outputs themselves: failed self-checks and impossible values."""

import json

import pytest
from tests.test_agent_evidence import RecordedClient, default_policy  # noqa: F401 (autouse)

from agent import loop
from agent.loop import AgentSolver
from agent.red_flags import red_flags


@pytest.mark.parametrize(
    "data, flagged",
    [
        ({"mc_validates": False}, "mc_validates = false"),
        ({"checks": [{"name": "parity", "passed": False}]}, "(parity)"),
        ({"summary": {"all_checks_passed": "FAIL"}}, "all_checks_passed"),
        ({"call_price": -0.35}, "negative value"),
        ({"rows": [{"default_probability": 1.2}]}, "probability above 1"),
        # Honest data properties and signed quantities are not red flags.
        ({"feller_satisfied": False, "reject_null": False}, None),
        ({"price_change": -1.0, "put_delta": -0.4, "log_vol": -2.0, "pnl": -5}, None),
        ({"mc_validates": True, "call_price": 10.45, "probability": 0.3}, None),
    ],
)
def test_json_red_flags(tmp_path, data, flagged):
    (tmp_path / "r.json").write_text(json.dumps(data))
    found = "\n".join(red_flags(tmp_path))
    assert flagged in found if flagged else not found


def test_table_red_flags(tmp_path):
    (tmp_path / "prices.csv").write_text("strike,price,vega\n90,1.2,-0.1\n100,-0.4,0.2\n")
    (tmp_path / "checks.csv").write_text("check,within_tolerance\na,True\nb,False\n")
    found = "\n".join(red_flags(tmp_path))
    assert "'price': negative value" in found and "1 of 2 rows" in found
    assert "'within_tolerance': false in 1 of 2 rows" in found
    assert "vega" not in found


def test_harness_files_are_ignored(tmp_path):
    (tmp_path / "reward.json").write_text('{"passed": false}')
    assert red_flags(tmp_path) == []


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
    (task / "instruction.md").write_text("Price the option; write /output/r.json.")
    return AgentSolver(task, tmp_path / "out", client=client)


def test_failed_self_check_drives_a_repair_despite_a_model_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "RED_FLAGS", True)
    client = RecordedClient(
        output({"price": 9.1, "mc_validates": False}),
        output({"price": 10.4, "mc_validates": True}),
        "VERDICT: PASS",
    )
    solver = solver_for(tmp_path, client)
    assert solver.run()
    assert json.loads((solver.out_dir / "r.json").read_text())["price"] == 10.4
    assert [kwargs["phase"] for _, kwargs in client.requests] == ["generate", "repair", "review"]
    assert "mc_validates = false" in client.requests[1][0]
    assert "SELF-CHECKS" in client.requests[0][0]


def test_unfixable_flag_keeps_the_best_outputs_after_bounded_repairs(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "RED_FLAGS", True)
    monkeypatch.setattr(loop, "MAX_GUARD_REPAIRS", 1)
    client = RecordedClient(
        output({"price": 9.1, "mc_validates": False}),
        output({"price": -1.0, "mc_validates": False}),
    )
    solver = solver_for(tmp_path, client)
    assert solver.run()
    # The repair added a second violation, so the first version is retained.
    assert json.loads((solver.out_dir / "r.json").read_text())["price"] == 9.1
    assert len(client.requests) == 2


def test_red_flags_off_by_default_leaves_the_prompt_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "RED_FLAGS", False)
    client = RecordedClient(output({"price": 9.1, "mc_validates": False}), "VERDICT: PASS")
    assert solver_for(tmp_path, client).run()
    assert "SELF-CHECKS" not in client.requests[0][0]
    assert [kwargs["phase"] for _, kwargs in client.requests] == ["generate", "review"]
