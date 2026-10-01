"""General contracts and identities using synthetic specifications, not benchmark answers."""

import json

import pytest

from agent.guardrails import derive_guardrails, numerical_guidance


@pytest.mark.parametrize(
    "value, expected",
    [
        ([{"trades": []}], "JSON object"),
        ({"trades": {"pnl": 2}}, "JSON array"),
        ({"trades": [{"pnl": "2"}]}, "finite JSON number"),
        ({"trades": [{"pnl": float("nan")}]}, "finite JSON number"),
        ({"trades": [{"pnl": 2}]}, None),
        ({"trades": []}, None),
    ],
)
def test_json_container_and_nested_numeric_contracts(tmp_path, value, expected):
    instruction = '### activity.json\n```json\n{"trades": [{"pnl": <float>}]}\n```'
    guard = derive_guardrails(instruction, ["activity.json"])
    (tmp_path / "activity.json").write_text(json.dumps(value))
    problems = guard.findings(tmp_path)
    assert any(expected in p for p in problems) if expected else not problems


def test_dynamic_keys_validate_values_without_requiring_placeholder_names(tmp_path):
    instruction = '### values.json\n```json\n{"rates": {"<CCY>": <float>}}\n```'
    guard = derive_guardrails(instruction, ["values.json"])
    path = tmp_path / "values.json"
    path.write_text('{"rates": {"XYZ": 0.1, "ABC": 0.2}}')
    assert not guard.findings(tmp_path)
    path.write_text('{"rates": {"XYZ": "bad"}}')
    assert any("rates.XYZ" in p for p in guard.findings(tmp_path))


def test_pseudo_json_type_tokens_and_unknown_quoted_placeholders(tmp_path):
    instruction = '### values.json\n```json\n{"monthly": [float x 12], "method": "<method>"}\n```'
    guard = derive_guardrails(instruction, ["values.json"])
    path = tmp_path / "values.json"
    path.write_text('{"monthly": [0.1, 0.2], "method": "float"}')
    assert not guard.findings(tmp_path)
    path.write_text('{"monthly": [0.1, "bad"], "method": "float"}')
    assert any("monthly[1]" in p for p in guard.findings(tmp_path))


@pytest.mark.parametrize(
    "var, es, expected",
    [
        (-2, -3, "positive loss"),
        (3, 2, "ES must be >= VaR"),
        ("two", 3, "numeric loss"),
        (2, 3, None),
    ],
)
def test_positive_loss_tail_invariants(tmp_path, var, es, expected):
    instruction = (
        "Compute VaR and ES. Dollar amounts are positive numbers representing potential losses.\n"
        "### risk.csv\nColumns: `var_dollar, es_dollar`.\n"
    )
    guard = derive_guardrails(instruction, ["risk.csv"])
    (tmp_path / "risk.csv").write_text(f"var_dollar,es_dollar\n{var},{es}\n")
    problems = guard.findings(tmp_path)
    assert any(expected in p for p in problems) if expected else not problems


def test_return_tail_and_negative_shifted_losses_do_not_get_positive_loss_rule(tmp_path):
    path = tmp_path / "risk.csv"
    path.write_text("var_estimate,es_estimate\n-3,-2\n")
    instruction = "Estimate VaR and ES on the loss distribution.\n### risk.csv\nColumns: `var_estimate, es_estimate`."
    assert not derive_guardrails(instruction, ["risk.csv"]).findings(tmp_path)
    path.write_text("var_estimate,es_estimate\n-2,-3\n")
    assert not derive_guardrails("Report lower-tail return VaR and ES.", ["risk.csv"]).findings(
        tmp_path
    )


def test_fx_quote_orientation_invariant(tmp_path):
    instruction = (
        'FX valuation.\n### quotes.json\n```json\n{"<PAIR>": {"bid": <float>, "ask": <float>}}\n```'
    )
    (tmp_path / "quotes.json").write_text('{"X/Y": {"bid": 2.1, "ask": 2.0}}')
    assert any(
        "bid > ask" in p for p in derive_guardrails(instruction, ["quotes.json"]).findings(tmp_path)
    )


def test_numerical_guidance_selects_identities_by_specification_concepts():
    fx = numerical_guidance("FX forward and cross rates")
    assert "pip multiplier" in fx and "Reciprocal bid=1/ask" in fx
    assert "Kirk" not in fx and "Fitted-normal" not in fx
    options = numerical_guidance("Kirk spread options and Margrabe")
    assert "F2/(F2+K)" in options and "At K=0 Kirk equals Margrabe" in options
    assert "bond" not in options


@pytest.mark.parametrize(
    "description",
    [
        "Columns: `date` (`YYYY-MM-DD`), `model` (literal `A`/`B`/`C`), `score` (numeric).",
        """\n| Column | Meaning |
|---|---|
| `date` (`YYYY-MM-DD`) | Observation date |
| `model` (`A`, `B`, `C`) | Fitted model label |
| `score` | Numeric result |\n""",
    ],
)
def test_column_description_formats_and_values_are_not_required_columns(tmp_path, description):
    instruction = "### metrics.csv\n" + description
    (tmp_path / "metrics.csv").write_text("date,model,score\n2020-01-01,A,0.5\n")
    assert not derive_guardrails(instruction, ["metrics.csv"]).findings(tmp_path)
