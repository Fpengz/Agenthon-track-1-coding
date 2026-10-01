"""Instruction-derived contracts and transferable numerical checks, never answer keys.

Only explicit output shapes and loss-tail identities block acceptance. Domain guidance
helps the model select computations to test; it is not an alternate task solver.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any

from agent.spec import DeliverableSpec, deliverable_section, parse_spec, spec_problems

_UNKNOWN = "__contract_unknown__"
_NUMBER = "__contract_number__"
_INTEGER = "__contract_integer__"
_DYNAMIC = "__contract_dynamic__"
_MAX_BYTES = 20_000_000
_MAX_FINDINGS = 25


def _json_template(section: str) -> Any:
    block = re.search(r"```json\s*\n(.*?)```", section, re.S)
    if not block:
        return None
    text = block.group(1)
    text = re.sub(r'"<[^"<>\n]*>"\s*:', json.dumps(_DYNAMIC) + ":", text)
    text = re.sub(r'"<[^"<>\n]*>"', json.dumps(_UNKNOWN), text)

    def placeholder(match: re.Match[str]) -> str:
        word = match.group()[1:-1].strip().split(",")[0].lower()
        return json.dumps(
            _INTEGER
            if word in {"int", "integer"}
            else _NUMBER
            if word in {"float", "double", "number", "decimal"}
            else _UNKNOWN
        )

    text = re.sub(r"<[^<>\n]*>", placeholder, text)
    text = re.sub(r"//[^\n]*", "", text)
    # Ellipses in objects omit more fields; bare type tokens in pseudo-JSON describe values.
    text = re.sub(r",\s*\.\.\.\s*(?=[}\]])", "", text)
    # Replace pseudo-JSON types outside string literals only.
    parts = re.split(r'("(?:[^"\\]|\\.)*")', text)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(
            r"\b(float|int)(?:\s+x\s*\d+)?\b",
            lambda m: json.dumps(_INTEGER if m.group(1) == "int" else _NUMBER),
            parts[i],
        )
    text = "".join(parts)
    text = text.replace("...", json.dumps(_UNKNOWN))
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _shape_problems(template: Any, value: Any, path: str, found: list[str]) -> None:
    if len(found) >= _MAX_FINDINGS or template is None or template == _UNKNOWN:
        return
    if isinstance(template, dict):
        if not isinstance(value, dict):
            found.append(f"{path}: must be a JSON object, got {type(value).__name__}")
            return
        for key, child in template.items():
            if key == _DYNAMIC:
                for actual_key, actual_value in list(value.items())[:200]:
                    if actual_key not in template:
                        _shape_problems(child, actual_value, f"{path}.{actual_key}", found)
            elif key not in value:
                found.append(f"{path}: missing required key {key!r}")
            else:
                _shape_problems(child, value[key], f"{path}.{key}", found)
    elif isinstance(template, list):
        if not isinstance(value, list):
            found.append(f"{path}: must be a JSON array, got {type(value).__name__}")
        elif len(template) == 1:
            for i, item in enumerate(value[:200]):
                _shape_problems(template[0], item, f"{path}[{i}]", found)
    elif template in (_NUMBER, _INTEGER) or (
        isinstance(template, (int, float)) and not isinstance(template, bool)
    ):
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not numeric or not math.isfinite(value):
            found.append(f"{path}: must be a finite JSON number, got {value!r}")
        elif template == _INTEGER and value != int(value):
            found.append(f"{path}: must be an integer, got {value!r}")
    elif isinstance(template, bool) and not isinstance(value, bool):
        found.append(f"{path}: must be a JSON boolean, got {value!r}")


def _risk_pairs(names: list[str]) -> list[tuple[str, str]]:
    """VaR/ES columns with the same suffix, including dollar/estimate/level labels."""
    by_lower = {name.lower(): name for name in names}
    pairs = []
    for name in names:
        lowered = name.lower()
        if lowered == "var" or lowered.startswith("var_"):
            es = by_lower.get("es" + lowered[3:])
            if es:
                pairs.append((name, es))
    return pairs


def _risk_values(var: Any, es: Any, path: str, positive: bool) -> list[str]:
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (var, es)):
        return [f"{path}: VaR and ES must be numeric loss amounts, got {var!r}, {es!r}"]
    if not all(math.isfinite(v) for v in (var, es)):
        return [f"{path}: VaR/ES are non-finite: {var!r}, {es!r}"]
    found = []
    if positive and (var <= 0 or es <= 0):
        found.append(f"{path}: specification requires positive loss amounts; VaR={var}, ES={es}")
    if es < var - 1e-8 * max(1, abs(var), abs(es)):
        found.append(f"{path}: upper-tail loss ES must be >= VaR; VaR={var}, ES={es}")
    return found


def _json_invariants(value: Any, path: str, guard: OutputGuardrails, found: list[str]) -> None:
    if len(found) >= _MAX_FINDINGS:
        return
    if isinstance(value, dict):
        if guard.loss_tail:
            for var, es in _risk_pairs(list(value)):
                found.extend(_risk_values(value[var], value[es], path, guard.positive_losses))
        if guard.fx_quotes and all(k in value for k in ("bid", "ask")):
            bid, ask = value["bid"], value["ask"]
            if all(isinstance(v, (int, float)) for v in (bid, ask)) and bid > ask:
                found.append(f"{path}: FX quote has bid > ask ({bid} > {ask}); check reciprocals")
        for key, child in list(value.items())[:200]:
            _json_invariants(child, f"{path}.{key}", guard, found)
    elif isinstance(value, list):
        for i, child in enumerate(value[:200]):
            _json_invariants(child, f"{path}[{i}]", guard, found)


@dataclass
class OutputGuardrails:
    specs: list[DeliverableSpec]
    templates: dict[str, Any] = field(default_factory=dict)
    loss_tail: bool = False
    positive_losses: bool = False
    fx_quotes: bool = False
    guidance: str = ""

    def findings(self, out_dir: pathlib.Path) -> list[str]:
        found = spec_problems(self.specs, out_dir)
        for path in sorted(out_dir.rglob("*")):
            if not path.is_file() or path.stat().st_size > _MAX_BYTES:
                continue
            name = path.relative_to(out_dir).as_posix()
            try:
                if path.suffix.lower() == ".json" and name in self.templates:
                    value = json.loads(path.read_text())
                    _shape_problems(self.templates[name], value, name, found)
                    _json_invariants(value, name, self, found)
                elif self.loss_tail and path.suffix.lower() in {".csv", ".tsv"}:
                    import pandas as pd

                    frame = pd.read_csv(
                        path, sep="\t" if path.suffix.lower() == ".tsv" else ",", nrows=200_000
                    )
                    for var_col, es_col in _risk_pairs(list(frame.columns)):
                        for i, (var, es) in enumerate(zip(frame[var_col], frame[es_col])):
                            problems = _risk_values(
                                var, es, f"{name} row {i + 1}", self.positive_losses
                            )
                            found.extend(problems)
                            if problems:
                                break  # one concrete counterexample per column pair
            except Exception as exc:
                found.append(
                    f"{name}: contract check could not read output ({type(exc).__name__}: {exc})"
                )
        return list(dict.fromkeys(found))[:_MAX_FINDINGS]


def numerical_guidance(instruction: str) -> str:
    """Select compact domain identities by concepts in the specification, not unit IDs."""
    text = instruction.lower()
    rules = [
        "Build every requested computation and deliverable before polishing any one part. "
        "Read back the exact JSON nesting/keys and CSV numeric dtypes after writing.",
        "Track units through formulas (percent/decimal, return/loss, dollars/units, annual/daily). "
        "Use explicit pandas index alignment; assert matching labels and finite required results.",
        "Review each requested stage with an independent identity or calculation on actual outputs. "
        "A matching summary and source column checks aggregation only; also validate the source computation. "
        "Repair a measured discrepancy and rerun its check before declaring PASS.",
        "Use installed APIs: scipy.special.ndtr or scipy.stats.norm.cdf; numpy.trapezoid; "
        "pandas month-end frequency 'ME'. Keep module aliases distinct from local variables.",
    ]
    if re.search(r"\bfx\b|foreign exchange|cross rates", text):
        rules.append(
            "FX: A/B means B units per A. Reciprocal bid=1/ask, ask=1/bid; "
            "A/C=(A/B)*(B/C). Check both quote orientations and at least one cross. "
            "Forward points=(forward-spot)*the specified pip multiplier, before rounding. "
            "Apply the buy/sell sign to valuation AND attribution; convert quote PnL to USD once."
        )
    if re.search(r"\bvar\b|value.at.risk|expected shortfall", text):
        rules.append(
            "Risk: for returns use losses=-returns, upper-tail quantile(alpha), ES over that same tail. "
            "Loss ES>=VaR. Fitted-normal ES=mu+sigma*phi(z)/(1-alpha), z=norm.ppf(alpha); "
            "evaluate phi at standardized z. Check heavy-tail and shifted/scaled examples too. "
            "Follow the specified EWMA recursion/initialization, quantile convention, ddof and horizon. "
            "For KDE check CDF(VaR)≈alpha and tail mass≈1-alpha; preserve the required RNG draw order."
        )
    if "brinson" in text or "weights drift" in text:
        rules.append(
            "Attribution: keep weights indexed by sector throughout; map sector->ETF only when reading returns. "
            "Assert sum(beginning weights)=1 after each drift/rebalance and retain cash. "
            "Check allocation+selection+interaction equals monthly portfolio minus benchmark return; "
            "link January to the prior December and distinguish summed effects from compounded returns."
        )
    if "kirk" in text or "margrabe" in text:
        rules.append(
            "Spread options: Kirk uses weight b=F2/(F2+K), effective variance "
            "sigma1²+b²*sigma2²-2*rho*sigma1*sigma2*b. At K=0 Kirk equals Margrabe. "
            "Check that identity across correlations and compare actual prices to independently "
            "discounted Monte Carlo payoffs, accounting for approximation and sampling error."
        )
    if "immuniz" in text or "key rate duration" in text:
        rules.append(
            "Hedging: cash allocation/price=units; multiply per-bond dollar sensitivity by units, "
            "or normalized sensitivity by cash allocation. Check PV and each hedge constraint "
            "separately after optimization, plus bounds and solver success. Scale constraints for "
            "conditioning. Reprice par instruments and check fitted-curve residuals at every tenor."
        )
    return "### NUMERICAL VALIDATION:\n" + "\n".join(f"- {rule}" for rule in rules)


def derive_guardrails(instruction: str, deliverables: list[str]) -> OutputGuardrails:
    text = instruction.lower()
    risk = bool(re.search(r"\bvar\b|value.at.risk", text))
    loss_tail = risk and bool(
        re.search(r"loss distribution|loss data|potential losses|simulated loss", text)
    )
    positive = loss_tail and bool(
        re.search(
            r"positive numbers representing potential losses|(?:var|es)[^\n.]{0,70}\bpositive\b",
            text,
        )
    )
    templates = {}
    for name in deliverables:
        if pathlib.Path(name).suffix.lower() == ".json":
            template = _json_template(deliverable_section(instruction, name))
            if template is not None:
                templates[name] = template
    return OutputGuardrails(
        parse_spec(instruction, deliverables),
        templates,
        loss_tail,
        positive,
        bool(re.search(r"\bfx\b|foreign exchange|cross rates", text)),
        numerical_guidance(instruction),
    )
