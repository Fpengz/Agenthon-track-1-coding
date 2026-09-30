"""agent_skills: vetted helpers that generated scripts may import (``AGENT_SKILLS``).

This file is copied next to each generated script as ``agent_skills.py`` (standard library,
numpy and pandas only). It targets recurring failure sources: JSON deliverables with NaN/inf or
numpy types (invalid strict JSON), reading the many input formats, and routine performance
statistics that are easy to get subtly wrong (annualisation, drawdown sign). Every function is
unit-tested in tests/test_base_agent.py.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any

import numpy as np
import pandas as pd

SKILLS_SUMMARY = """Optional helper module `agent_skills` (importable; tested). Use it where it fits:
- write_json(path, obj): strict JSON; NaN/inf -> null; numpy/pandas scalars, arrays, Timestamps ok
- read_table(path, **kw) -> DataFrame for .csv/.tsv/.parquet/.pqt/.xlsx/.json/.jsonl
- to_datetime_index(df, column) -> df indexed by a sorted DatetimeIndex (column parsed)
- annualized_return(returns, periods_per_year), annualized_vol(returns, periods_per_year)
- sharpe_ratio(returns, periods_per_year, risk_free_per_period=0.0)
- max_drawdown(returns_or_equity, is_returns=True) -> negative fraction, e.g. -0.23
- cumulative_returns(returns) -> (1 + r).cumprod() - 1"""


def _plain(obj: Any) -> Any:
    """Recursively convert to JSON-safe Python values (NaN/inf become None)."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_plain(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Series, pd.Index)):
        return [_plain(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return None if pd.isna(obj) else pd.Timestamp(obj).isoformat()
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if obj is pd.NaT:
        return None
    return obj


def write_json(path: str | pathlib.Path, obj: Any, indent: int = 2) -> None:
    """Write strict JSON (no NaN/Infinity tokens), converting numpy/pandas values."""
    pathlib.Path(path).write_text(
        json.dumps(_plain(obj), indent=indent, allow_nan=False), encoding="utf-8"
    )


def read_table(path: str | pathlib.Path, **kwargs: Any) -> pd.DataFrame:
    """Read a table by file extension."""
    path = pathlib.Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", **kwargs)
    if suffix in {".parquet", ".pqt"}:
        return pd.read_parquet(path, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True, **kwargs)
    if suffix == ".json":
        return pd.read_json(path, **kwargs)
    raise ValueError(f"unsupported table format: {path.name}")


def to_datetime_index(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return ``df`` indexed by ``column`` parsed as datetimes, sorted ascending."""
    out = df.copy()
    out[column] = pd.to_datetime(out[column])
    return out.set_index(column).sort_index()


def _series(values: Any) -> pd.Series:
    return pd.Series(values, dtype=float).dropna()


def annualized_return(returns: Any, periods_per_year: int) -> float:
    """Geometric annualised return of periodic simple returns."""
    r = _series(returns)
    if r.empty:
        return float("nan")
    growth = float((1.0 + r).prod())
    return growth ** (periods_per_year / len(r)) - 1.0


def annualized_vol(returns: Any, periods_per_year: int) -> float:
    """Sample standard deviation (ddof=1) of periodic returns, scaled by sqrt(periods)."""
    r = _series(returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1)) * math.sqrt(periods_per_year)


def sharpe_ratio(returns: Any, periods_per_year: int, risk_free_per_period: float = 0.0) -> float:
    """Annualised Sharpe ratio of excess periodic returns (mean / sample std * sqrt(periods))."""
    excess = _series(returns) - risk_free_per_period
    if len(excess) < 2:
        return float("nan")
    std = float(excess.std(ddof=1))
    if std == 0.0:
        return float("nan")
    return float(excess.mean()) / std * math.sqrt(periods_per_year)


def cumulative_returns(returns: Any) -> pd.Series:
    """Compounded cumulative return path: (1 + r).cumprod() - 1."""
    return (1.0 + pd.Series(returns, dtype=float).fillna(0.0)).cumprod() - 1.0


def max_drawdown(values: Any, is_returns: bool = True) -> float:
    """Maximum drawdown as a NEGATIVE fraction (e.g. -0.23), from returns or an equity curve."""
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return float("nan")
    equity = (1.0 + series).cumprod() if is_returns else series
    if is_returns:
        equity = pd.concat([pd.Series([1.0]), equity], ignore_index=True)  # include the start
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())
