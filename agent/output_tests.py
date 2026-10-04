"""Generic output tests for the failure kinds that recur in recorded runs.

Wrong numbers are the main failure (about 45 of 86 units per run), and several frequent kinds
are visible in the deliverables without knowing the answer. A failure analysis over 18 full
runs (1,164 failed unit runs) found these recurring kinds:

- the output reports its own error as large: a repricing residual of thousands of bps, a
  parity error of 56, a model-vs-benchmark gap of 13, a relative error of 0.79;
- a no-arbitrage relationship is broken: an American price below its European counterpart,
  a negative price;
- degenerate results: one value repeated for every instrument (a computation that ignores its
  inputs or never left its initial guess), a computed column that is constant;
- a validation flag the task asked for reported false (``red_flags``);
- a rate reported in percent (2.09 for 209%) beside rates in decimals.

Each test returns ``TestFailure``s with the evidence and the usual causes; failures drive
repair requests and rank consensus candidates (``AGENT_OUTPUT_TESTS``). Tests never name a
unit or hold an expected value. Rule: every test must stay (near-)silent on checker-passed
outputs (see ``scripts`` in the commit that added this module for the audit).
"""

from __future__ import annotations

import json
import math
import pathlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from agent.red_flags import red_flags
from agent.review import HARNESS_FILES

_MAX_BYTES = 20_000_000
_MAX_EVIDENCE = 6


@dataclass(frozen=True)
class TestFailure:
    test: str
    evidence: list[str]
    causes: str

    def render(self) -> str:
        lines = "\n".join(f"  - {e}" for e in self.evidence[:_MAX_EVIDENCE])
        more = len(self.evidence) - _MAX_EVIDENCE
        tail = f"\n  - ... and {more} more" if more > 0 else ""
        return f"TEST FAILED: {self.test}\n{lines}{tail}\n  Usual causes: {self.causes}"


def _tokens(name: str) -> list[str]:
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(name))
    return [t for t in re.split(r"[^0-9a-zA-Z]+", name.lower()) if t]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


# ---------------------------------------------------------------------------------------------
# Loading: every deliverable as named scalar leaves (JSON) and numeric columns (tables).


@dataclass
class _Outputs:
    leaves: list[tuple[str, str, float]]  # (path, key, value) for JSON numeric leaves
    siblings: list[tuple[str, dict[str, float]]]  # (path, {key: value}) per JSON object
    records: list[tuple[str, str, list[float]]]  # (path, key, values) across a list of objects
    columns: list[tuple[str, str, list[float], bool]]  # (file, column, values, integer dtype)
    tables: list[tuple[str, Any]]  # (file, DataFrame)


def _walk(value: Any, path: str, out: _Outputs) -> None:
    if isinstance(value, dict):
        numbers: dict[str, float] = {}
        for key, child in list(value.items())[:2000]:
            number = _number(child)
            if number is not None:
                out.leaves.append((f"{path}.{key}", str(key), number))
                numbers[str(key)] = number
            else:
                _walk(child, f"{path}.{key}", out)
        if numbers:
            out.siblings.append((path, numbers))
    elif isinstance(value, list):
        items = value[:5000]
        if items and all(isinstance(i, dict) for i in items):
            by_key: dict[str, list[float]] = defaultdict(list)
            for item in items:
                for key, child in item.items():
                    number = _number(child)
                    if number is not None:
                        by_key[str(key)].append(number)
            for key, values in by_key.items():
                if len(values) == len(items):
                    out.records.append((path, key, values))
        for i, child in enumerate(items[:200]):
            _walk(child, f"{path}[{i}]", out)


def load_outputs(out_dir: pathlib.Path) -> _Outputs:
    import pandas as pd

    out = _Outputs([], [], [], [], [])
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path.name in HARNESS_FILES:
            continue
        name = path.relative_to(out_dir).as_posix()
        suffix = path.suffix.lower()
        try:
            if path.stat().st_size > _MAX_BYTES:
                continue
            if suffix == ".json":
                _walk(json.loads(path.read_text(encoding="utf-8")), name, out)
                continue
            if suffix in {".parquet", ".pqt"}:
                frame = pd.read_parquet(path)
            elif suffix in {".csv", ".tsv"}:
                frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", nrows=200_000)
            else:
                continue
        except Exception:  # unreadable outputs are reported by the other output checks
            continue
        out.tables.append((name, frame))
        for column in frame.columns:
            series = frame[column]
            if series.dtype == bool or not pd.api.types.is_numeric_dtype(series):
                continue
            values = [float(v) for v in series.dropna() if math.isfinite(float(v))]
            if values:
                out.columns.append(
                    (name, str(column), values, pd.api.types.is_integer_dtype(series))
                )
    return out


# ---------------------------------------------------------------------------------------------
# Tests.

_ERROR_TOKENS = {"residual", "residuals", "error", "err", "rmse", "mae", "discrepancy"}
_REL_TOKENS = {"rel", "relative", "pct", "percent"}
_BPS_TOKENS = {"bp", "bps"}


def test_self_reported_errors(out: _Outputs) -> TestFailure | None:
    """A residual/error the output reports about itself that is far too large to be honest."""
    evidence = []
    for path, key, value in out.leaves:
        tokens = set(_tokens(key))
        if not tokens & _ERROR_TOKENS or "std" in tokens or "standard" in tokens:
            continue
        if tokens & {"tracking", "forecast", "prediction", "pricing", "fit", "model"}:
            continue  # statistics of a model fit, not a correctness check
        # bps residuals above 100, relative errors above 0.25, percent errors above 50 percent
        limit = (
            100.0
            if tokens & _BPS_TOKENS
            else 50.0
            if tokens & {"pct", "percent"}
            else 0.25
            if tokens & {"rel", "relative"}
            else None
        )
        if limit is not None and abs(value) > limit:
            evidence.append(f"{path} = {value:.6g}")
    if not evidence:
        return None
    return TestFailure(
        "self-reported errors are implausibly large",
        evidence,
        "a residual/error this large means the computation it checks is wrong, not that the "
        "tolerance is loose: wrong day-count or discounting, rates in percent instead of "
        "decimals, a curve/model solved against the wrong instrument definition, or the two "
        "sides of the comparison computed with different conventions. Recompute one instrument "
        "by hand from the inputs and find where the two sides diverge.",
    )


def _pair_key(tokens: list[str], word: str, other: str) -> tuple[str, ...] | None:
    if word not in tokens:
        return None
    return tuple(other if t == word else t for t in tokens)


def test_american_geq_european(out: _Outputs) -> TestFailure | None:
    """An American option is worth at least the matching European option."""
    evidence = []
    for path, numbers in out.siblings:
        for key, value in numbers.items():
            tokens = _tokens(key)
            twin = _pair_key(tokens, "american", "european") or _pair_key(tokens, "amer", "euro")
            if twin is None or {"premium", "diff", "difference", "ratio", "error"} & set(tokens):
                continue
            for other_key, other in numbers.items():
                if tuple(_tokens(other_key)) == twin and value < other - 1e-6 * max(1, abs(other)):
                    evidence.append(
                        f"{path}.{key} = {value:.6g} < {path}.{other_key} = {other:.6g}"
                    )
    for name, frame in out.tables:
        cols = {tuple(_tokens(c)): c for c in frame.columns}
        for tokens, column in cols.items():
            twin = _pair_key(list(tokens), "american", "european") or _pair_key(
                list(tokens), "amer", "euro"
            )
            if twin is None or twin not in cols:
                continue
            try:
                a = frame[column].astype(float)
                e = frame[cols[twin]].astype(float)
            except (TypeError, ValueError):
                continue
            bad = (a < e - 1e-6 * e.abs().clip(lower=1)).sum()
            if bad:
                evidence.append(f"{name}: {column} < {cols[twin]} in {int(bad)} of {len(a)} rows")
    if not evidence:
        return None
    return TestFailure(
        "American value below European value",
        evidence,
        "early exercise can only add value, so an American price below the European one is "
        "a pricing bug: wrong early-exercise comparison (intrinsic vs continuation), call/put "
        "payoff swapped, dividend yield or rate used with the wrong sign, or the two prices "
        "computed on different grids/parameters.",
    )


def test_repeated_values(out: _Outputs) -> TestFailure | None:
    """One non-trivial value repeated across different instruments, or a constant result column."""
    evidence = []
    for path, key, values in out.records:
        if len(values) >= 4 and _degenerate(values) and _result_name(key):
            evidence.append(f"{path}[*].{key} = {values[0]:.6g} for all {len(values)} items")
    for name, column, values, integer in out.columns:
        if len(values) >= 5 and not integer and _degenerate(values) and _result_name(column):
            evidence.append(f"{name} column {column!r} = {values[0]:.6g} in all {len(values)} rows")
    if not evidence:
        return None
    return TestFailure(
        "a computed result is the same for every instrument/row",
        evidence,
        "a quantity that should vary across instruments (strikes, maturities, assets) but is "
        "constant usually means the loop reuses one instrument's parameters, a solver never "
        "moved from its initial guess (or failed and a default was written), or a merge/"
        "broadcast assigned one value to all rows. Print the per-instrument inputs inside the "
        "loop and check that they change.",
    )


_RESULT_TOKENS = {
    "price", "value", "pv", "npv", "premium", "iv", "vol", "volatility", "skew", "var", "es",
    "cvar", "delta", "gamma", "vega", "theta", "rate", "yield", "spread", "return", "sharpe",
    "weight", "beta", "alpha", "prob", "probability", "pnl", "duration", "convexity",
}  # fmt: skip


# Input echoes that may legitimately repeat (one trade's entry price, a common strike).
_INPUT_TOKENS = {"entry", "exit", "initial", "strike", "spot", "notional", "input", "target"}


def _result_name(name: str) -> bool:
    tokens = set(_tokens(name))
    return bool(tokens & _RESULT_TOKENS) and not tokens & _INPUT_TOKENS


def _degenerate(values: list[float]) -> bool:
    first = values[0]
    if first == 0 or float(first).is_integer():
        return False  # zeros and integer codes are often legitimate constants
    return all(abs(v - first) <= 1e-12 * max(1.0, abs(first)) for v in values)


_RATE_TOKENS = {"rate", "yield", "zero", "par", "forward", "fwd", "libor", "ois", "sofr", "coupon"}


def test_percent_rates(out: _Outputs) -> TestFailure | None:
    """A rate in percent beside sibling rates in decimals (2.09 next to 0.04)."""
    evidence = []
    for path, numbers in out.siblings:
        rates = {
            k: v
            for k, v in numbers.items()
            if set(_tokens(k)) & _RATE_TOKENS
            and not set(_tokens(k)) & (_BPS_TOKENS | _REL_TOKENS | {"count", "n", "num", "log"})
        }
        small = [v for v in rates.values() if 0 < abs(v) < 0.5]
        if len(small) < 2:
            continue
        for key, value in rates.items():
            if 1.0 < abs(value) < 100:
                evidence.append(f"{path}.{key} = {value:.6g} (other rates here are decimals)")
    if not evidence:
        return None
    return TestFailure(
        "a rate looks like percent among decimal rates",
        evidence,
        "mixed units: one rate was taken from a percent-quoted input (or multiplied by 100) "
        "while the others are decimals, or an annualisation/compounding step was applied "
        "twice. Use one unit convention throughout, as the specification states.",
    )


def _option_side(tokens: set[str]) -> str | None:
    if tokens & {"call", "calls"} and not tokens & {"put", "puts"}:
        return "call"
    if tokens & {"put", "puts"} and not tokens & {"call", "calls"}:
        return "put"
    return None


def test_strike_monotonicity(out: _Outputs) -> TestFailure | None:
    """Call prices fall and put prices rise with the strike (same expiry and model)."""
    evidence = []
    for name, frame in out.tables:
        strike = next((c for c in frame.columns if set(_tokens(c)) & {"strike", "k"}), None)
        if strike is None:
            continue
        side_col = next(
            (c for c in frame.columns if set(_tokens(c)) & {"type", "option", "cp", "side"}),
            None,
        )
        groups = [
            c
            for c in frame.columns
            if c not in (strike, side_col)
            and set(_tokens(c)) & {"maturity", "expiry", "tenor", "t", "ttm", "model", "method"}
        ]
        for column in frame.columns:
            tokens = set(_tokens(column))
            if not tokens & {"price", "value", "premium"} or tokens & _INPUT_TOKENS:
                continue
            side = _option_side(tokens)
            subsets = []
            if side:
                subsets = [(side, frame)]
            elif side_col is not None:
                for label, part in frame.groupby(side_col):
                    side = _option_side(set(_tokens(str(label)))) or {
                        "c": "call",
                        "p": "put",
                    }.get(str(label).strip().lower())
                    if side:
                        subsets.append((side, part))
            for side, part in subsets:
                try:
                    parts = part.groupby(groups) if groups else [(None, part)]
                    for _, g in parts:
                        g = g[[strike, column]].astype(float).dropna().sort_values(strike)
                        g = g.drop_duplicates(strike)
                        if len(g) < 3:
                            continue
                        step = g[column].diff().dropna()
                        tol = 1e-6 * max(1.0, float(g[column].abs().max()))
                        bad = int((step > tol).sum() if side == "call" else (step < -tol).sum())
                        if bad:
                            direction = "rises" if side == "call" else "falls"
                            evidence.append(
                                f"{name}: {side} {column} {direction} with {strike} at {bad} of "
                                f"{len(step)} strike steps"
                            )
                except (TypeError, ValueError, KeyError):
                    continue
    if not evidence:
        return None
    return TestFailure(
        "option prices are not monotone in the strike",
        evidence,
        "for one expiry and model, call prices must fall and put prices rise as the strike "
        "increases. A violation usually means call/put labels or payoffs are swapped, strikes "
        "and prices are misaligned (sorting or merging one without the other), or the "
        "numerical method (grid, tree, simulation, integration) is unstable for some strikes.",
    )


def test_probability_rows(out: _Outputs) -> TestFailure | None:
    """Rows of a transition/probability matrix sum to 1 (or 100 when in percent)."""
    evidence = []
    for name, frame in out.tables:
        if not set(_tokens(name)) & {"transition", "migration", "probability", "probabilities"}:
            continue
        numeric = frame.select_dtypes("number")
        numeric = numeric[[c for c in numeric.columns if not set(_tokens(c)) & {"count", "n"}]]
        if numeric.shape[1] < 3 or numeric.empty:
            continue
        values = numeric.to_numpy(dtype=float)
        if (values < -1e-9).any():
            continue  # generator matrices have negative diagonals
        sums = values.sum(axis=1)
        scale = 100.0 if sums.max() > 1.5 else 1.0
        bad = [i for i, s in enumerate(sums) if abs(s - scale) > 1e-3 * scale and s > 0]
        if bad:
            evidence.append(
                f"{name}: {len(bad)} of {len(sums)} rows do not sum to {scale:g} "
                f"(row {bad[0] + 1} sums to {sums[bad[0]]:.6g})"
            )
    if not evidence:
        return None
    return TestFailure(
        "probability rows do not sum to one",
        evidence,
        "each row of a transition/probability matrix is a distribution. A row that does not "
        "sum to 1 means counts were normalised by the wrong total (all rows instead of the "
        "row, or before dropping/merging states), a state is missing from the columns, or "
        "annualising/exponentiating a generator went wrong.",
    )


TESTS = (
    test_self_reported_errors,
    test_american_geq_european,
    test_repeated_values,
    test_percent_rates,
    test_strike_monotonicity,
    test_probability_rows,
)


def output_test_failures(out_dir: pathlib.Path) -> list[str]:
    """Rendered failures for a clean run's deliverables (empty when every test passes)."""
    try:
        out = load_outputs(out_dir)
    except Exception:
        return []
    failures = [f.render() for test in TESTS if (f := test(out))]
    return failures + red_flags(out_dir)
