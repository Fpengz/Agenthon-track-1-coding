"""Staged generation: plan short step scripts, debug them in order, hand the combination over."""

import json
import subprocess
import sys

import pytest
from tests.test_agent_evidence import RecordedClient, default_policy  # noqa: F401 (autouse)

from agent import loop, staged
from agent.loop import AgentSolver


def plan(*steps):
    return "```json\n" + json.dumps({"steps": [dict(s) for s in steps]}) + "\n```"


def py(code):
    return f"```python\n{code}\n```"


STEP1 = """import os, pathlib
stage = pathlib.Path(os.environ["STAGE_DIR"])
(stage / "x.txt").write_text("2.5")"""
STEP2 = """import os, pathlib, json
x = float((pathlib.Path(os.environ["STAGE_DIR"]) / "x.txt").read_text())
(pathlib.Path(os.environ["OUTPUT_DIR"]) / "r.json").write_text(json.dumps({"price": x}))"""
PLAN = plan(
    {"name": "load", "goal": "read x", "writes": ["STAGE_DIR/x.txt"]},
    {"name": "write", "goal": "write r.json", "writes": ["OUTPUT_DIR/r.json"]},
)


def test_parse_plan_assigns_missing_deliverables_to_the_last_step():
    steps = staged.parse_plan(
        "thinking...\n" + plan({"name": "a", "goal": "g"}, {"name": "b", "goal": "h"}),
        ["r.json", "t.csv"],
    )
    assert [s.name for s in steps] == ["a", "b"]
    assert steps[-1].writes == ["OUTPUT_DIR/r.json", "OUTPUT_DIR/t.csv"]


@pytest.mark.parametrize("text", ["no json", plan({"name": "only"}), "```json\n{bad\n```"])
def test_unusable_plans_are_rejected(text):
    assert staged.parse_plan(text, ["r.json"]) is None


def test_combined_script_runs_the_steps_in_one_process(tmp_path):
    steps = staged.parse_plan(PLAN, ["r.json"])
    script = tmp_path / "s.py"
    script.write_text(staged.combine([STEP1, STEP2], steps))
    env = {"OUTPUT_DIR": str(tmp_path), "PATH": ""}
    subprocess.run([sys.executable, str(script)], check=True, env=env)
    assert json.loads((tmp_path / "r.json").read_text()) == {"price": 2.5}


def solver_for(tmp_path, client):
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("Write /output/r.json with price equal to 2.5.")
    return AgentSolver(task, tmp_path / "out", client=client)


def test_staged_steps_are_debugged_then_handed_to_the_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "STAGED", True)
    client = RecordedClient(
        PLAN,
        py(STEP1),
        py("raise ValueError('step two broke')"),  # step 2 fails once, then is repaired
        py(STEP2),
        "VERDICT: PASS",
    )
    solver = solver_for(tmp_path, client)
    assert solver.run()
    assert json.loads((solver.out_dir / "r.json").read_text()) == {"price": 2.5}
    phases = [kwargs["phase"] for _, kwargs in client.requests]
    assert phases == ["plan", "generate", "generate", "repair", "review"]
    assert "step two broke" in client.requests[3][0]
    assert "STAGE_DIR/x.txt" in client.requests[1][0]  # the step sees the plan
    assert "# ===== step 2: write =====" in solver.accepted_code


def test_unusable_plan_falls_back_to_one_script(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "STAGED", True)
    client = RecordedClient(
        "no plan here",
        py(
            STEP1.replace("STAGE_DIR", "OUTPUT_DIR")
            + "\n"
            + STEP2.replace('os.environ["STAGE_DIR"]', 'os.environ["OUTPUT_DIR"]')
        ),
        "VERDICT: PASS",
    )
    solver = solver_for(tmp_path, client)
    assert solver.run()
    assert [kwargs["phase"] for _, kwargs in client.requests] == ["plan", "generate", "review"]


def test_unresolved_step_is_repaired_by_the_ordinary_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "STAGED", True)
    monkeypatch.setattr(loop, "STAGED_STEP_REPAIRS", 0)
    client = RecordedClient(
        PLAN,
        py(STEP1),
        py("raise ValueError('still broken')"),
        # The loop runs the combined script, sees the failure and asks for a repair.
        py(staged.COMBINED_HEADER + STEP1 + "\n" + STEP2),
        "VERDICT: PASS",
    )
    solver = solver_for(tmp_path, client)
    assert solver.run()
    assert json.loads((solver.out_dir / "r.json").read_text()) == {"price": 2.5}
    phases = [kwargs["phase"] for _, kwargs in client.requests]
    assert phases == ["plan", "generate", "generate", "repair", "review"]
    assert "still broken" in client.requests[3][0]


def test_staged_off_by_default(tmp_path):
    assert loop.STAGED is False


def test_step_cut_off_while_reasoning_is_retried_without_thinking(tmp_path, monkeypatch):
    from agent.client import ChatResult

    monkeypatch.setattr(loop, "STAGED", True)
    client = RecordedClient(
        PLAN,
        ChatResult("long reasoning that never reaches code " * 50, "length"),
        py(STEP1),
        py(STEP2),
        "VERDICT: PASS",
    )
    solver = solver_for(tmp_path, client)
    assert solver.run()
    kwargs = [k for _, k in client.requests]
    assert "extra_body" not in kwargs[1]
    assert kwargs[2]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "extra_body" not in kwargs[3]  # the next step reasons again


def test_plan_allows_up_to_eight_small_steps():
    steps = [{"name": f"s{i}", "goal": "g"} for i in range(8)]
    assert len(staged.parse_plan(plan(*steps), ["r.json"])) == 8
    assert staged.parse_plan(plan(*steps, {"name": "s9"}), ["r.json"]) is None


def test_steps_with_many_deliverables_are_split():
    text = plan(
        {"name": "a", "goal": "load", "writes": ["STAGE_DIR/x.csv"]},
        {
            "name": "b",
            "goal": "snapshot",
            "writes": ["STAGE_DIR/y.csv"] + [f"OUTPUT_DIR/f{i}.json" for i in range(5)],
        },
    )
    steps = staged.parse_plan(text, [f"f{i}.json" for i in range(5)])
    assert [s.name for s in steps] == ["a", "b_1", "b_2", "b_3"]
    assert steps[1].writes == ["STAGE_DIR/y.csv", "OUTPUT_DIR/f0.json", "OUTPUT_DIR/f1.json"]
    assert steps[3].writes == ["OUTPUT_DIR/f4.json"]
    assert "PART 3 of 3: write only f4.json" in steps[3].goal


def test_step_cut_off_mid_script_is_continued(tmp_path, monkeypatch):
    from agent.client import ChatResult

    monkeypatch.setattr(loop, "STAGED", True)
    head, tail = STEP2.split("\n", 1)
    client = RecordedClient(
        PLAN,
        py(STEP1),
        ChatResult(f"```python\n{head}\n", "length"),
        py(tail),
        "VERDICT: PASS",
    )
    solver = solver_for(tmp_path, client)
    assert solver.run()
    assert json.loads((solver.out_dir / "r.json").read_text()) == {"price": 2.5}
    assert [k["phase"] for _, k in client.requests] == [
        "plan",
        "generate",
        "generate",
        "continue",
        "review",
    ]
