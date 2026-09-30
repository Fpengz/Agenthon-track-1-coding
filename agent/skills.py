"""agent_skills: vetted helpers that generated scripts may import (``AGENT_SKILLS``).

This file is copied next to each generated script as ``agent_skills.py`` (standard library,
numpy, pandas, scipy). Two layers:

- utilities: strict JSON writer, table reader, date indexing;
- domain reference implementations for the calculation families the tasks ask for most
  (Black-Scholes and Greeks, bond maths, VaR/ES, GBM Monte Carlo, performance statistics).

The first version offered only generic utilities: 134 of 141 scripts used ``write_json`` and
almost none used anything else, because the helpers did not address where answers go wrong.
The reference implementations are general-purpose textbook formulas tested against analytic
values (never against a unit's checker); conventions are explicit arguments. They serve the
solution and, more importantly, the verifier, which needs numbers that do not share the
solution's misconceptions. Every function is unit-tested in tests/test_base_agent.py.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any

import numpy as np
import pandas as pd


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


# --------------------------------------------------------------------------------------------
# Domain reference implementations. Textbook formulas, tested against published analytic
# values (never against any unit's checker). Every convention is an explicit argument: the
# TASK SPECIFICATION decides the convention, the helper only removes arithmetic mistakes.
# --------------------------------------------------------------------------------------------


def _norm_cdf(x: Any) -> Any:
    from scipy.stats import norm

    return norm.cdf(x)


def _norm_pdf(x: Any) -> Any:
    from scipy.stats import norm

    return norm.pdf(x)


def bs_price(S, K, T, r, sigma, option_type: str = "call", q: float = 0.0):
    """Black-Scholes(-Merton) European price. T in years; r, q continuously compounded; sigma
    annualised decimal (0.20 = 20%). Works on scalars or numpy arrays."""
    S, K, T, sigma = (np.asarray(v, dtype=float) for v in (S, K, T, sigma))
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    call = S * np.exp(-q * T) * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2)
    if option_type.lower().startswith("c"):
        return call
    return call - S * np.exp(-q * T) + K * np.exp(-r * T)  # put-call parity


def bs_greeks(S, K, T, r, sigma, option_type: str = "call", q: float = 0.0) -> dict[str, Any]:
    """Black-Scholes Greeks. Units: delta per 1.0 of S; gamma per 1.0 of S squared; vega per 1.0
    of sigma (divide by 100 for "per vol point"); theta PER YEAR (divide by 365 or 252 for per
    day -- the specification says which); rho per 1.0 of r (divide by 100 for "per 1%")."""
    S, K, T, sigma = (np.asarray(v, dtype=float) for v in (S, K, T, sigma))
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    is_call = option_type.lower().startswith("c")
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)
    common_theta = -S * disc_q * _norm_pdf(d1) * sigma / (2 * sqrt_t)
    if is_call:
        delta = disc_q * _norm_cdf(d1)
        theta = common_theta - r * K * disc_r * _norm_cdf(d2) + q * S * disc_q * _norm_cdf(d1)
        rho = K * T * disc_r * _norm_cdf(d2)
    else:
        delta = -disc_q * _norm_cdf(-d1)
        theta = common_theta + r * K * disc_r * _norm_cdf(-d2) - q * S * disc_q * _norm_cdf(-d1)
        rho = -K * T * disc_r * _norm_cdf(-d2)
    return {
        "delta": delta,
        "gamma": disc_q * _norm_pdf(d1) / (S * sigma * sqrt_t),
        "vega": S * disc_q * _norm_pdf(d1) * sqrt_t,
        "theta": theta,
        "rho": rho,
    }


def implied_vol(price, S, K, T, r, option_type: str = "call", q: float = 0.0) -> float:
    """Black-Scholes implied volatility by root finding; NaN if the price admits none."""
    from scipy.optimize import brentq

    def gap(sigma: float) -> float:
        return float(bs_price(S, K, T, r, sigma, option_type, q)) - float(price)

    try:
        return float(brentq(gap, 1e-6, 10.0, xtol=1e-10, maxiter=200))
    except ValueError:
        return float("nan")


def bond_price(face: float, coupon_rate: float, ytm: float, years: float, freq: int = 2) -> float:
    """Price of a fixed-coupon bullet bond. coupon_rate and ytm are ANNUAL decimals compounded
    `freq` times a year; `years` to maturity must give a whole number of periods."""
    n = int(round(years * freq))
    coupon, y = face * coupon_rate / freq, ytm / freq
    times = np.arange(1, n + 1)
    return float(np.sum(coupon / (1 + y) ** times) + face / (1 + y) ** n)


def bond_ytm(price: float, face: float, coupon_rate: float, years: float, freq: int = 2) -> float:
    """Annual yield to maturity (compounded `freq` times a year) that reprices the bond."""
    from scipy.optimize import brentq

    return float(
        brentq(
            lambda y: bond_price(face, coupon_rate, y, years, freq) - price, -0.99, 5.0, xtol=1e-12
        )
    )


def bond_duration_convexity(
    face: float, coupon_rate: float, ytm: float, years: float, freq: int = 2
) -> dict[str, float]:
    """Macaulay duration (years), modified duration (= Macaulay / (1 + ytm/freq)) and convexity
    (years^2, the second-derivative measure: dP ~ -D_mod*dy*P + 0.5*C*dy^2*P)."""
    n = int(round(years * freq))
    coupon, y = face * coupon_rate / freq, ytm / freq
    k = np.arange(1, n + 1)
    flows = np.full(n, coupon)
    flows[-1] += face
    pv = flows / (1 + y) ** k
    price = float(pv.sum())
    macaulay = float((k / freq * pv).sum() / price)
    convexity = float((k * (k + 1) * pv).sum() / (price * (1 + y) ** 2 * freq**2))
    return {
        "price": price,
        "macaulay_duration": macaulay,
        "modified_duration": macaulay / (1 + y),
        "convexity": convexity,
    }


def historical_var_es(
    returns: Any, alpha: float = 0.99, method: str = "linear"
) -> dict[str, float]:
    """Historical VaR and Expected Shortfall at confidence `alpha`, as POSITIVE losses.
    VaR = -quantile(returns, 1 - alpha) with numpy's `method` (linear, lower, higher, ...);
    ES = -mean of the returns at or below that quantile. Pass P&L instead of returns for
    currency amounts."""
    r = np.asarray(pd.Series(returns, dtype=float).dropna())
    quantile: Any = np.quantile  # numpy's stub wants a literal method name
    cutoff = float(quantile(r, 1 - alpha, method=method))
    tail = r[r <= cutoff]
    return {"var": -cutoff, "es": -float(tail.mean()) if tail.size else -cutoff}


def parametric_var_es(mu: float, sigma: float, alpha: float = 0.99) -> dict[str, float]:
    """Normal (variance-covariance) VaR and ES at confidence `alpha`, as POSITIVE losses for a
    return with mean `mu` and standard deviation `sigma` over the horizon."""
    from scipy.stats import norm

    z = norm.ppf(1 - alpha)
    return {
        "var": float(-(mu + sigma * z)),
        "es": float(-(mu - sigma * norm.pdf(z) / (1 - alpha))),
    }


def gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    steps: int,
    n_paths: int,
    seed: int | None = None,
    q: float = 0.0,
    antithetic: bool = False,
) -> np.ndarray:
    """Risk-neutral geometric Brownian motion paths, shape (n_paths, steps + 1), exact
    log-normal stepping, `numpy.random.default_rng(seed)`. With antithetic=True the second half
    of the paths mirrors the first (n_paths must be even)."""
    rng = np.random.default_rng(seed)
    dt = T / steps
    half = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal((half, steps))
    if antithetic:
        z = np.vstack([z, -z])
    increments = (r - q - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((z.shape[0], 1)), np.cumsum(increments, axis=1)], axis=1)
    return S0 * np.exp(log_paths)


def performance_summary(
    returns: Any, periods_per_year: int, risk_free_per_period: float = 0.0, ddof: int = 1
) -> dict[str, float]:
    """Common performance statistics of periodic SIMPLE returns. annualized_return is
    geometric; annualized_vol = std(ddof) * sqrt(periods); sharpe uses the arithmetic mean of
    excess returns; max_drawdown is negative. Check each against the specification's formula."""
    r = _series(returns)
    excess = r - risk_free_per_period
    std = float(r.std(ddof=ddof)) if len(r) > ddof else float("nan")
    ex_std = float(excess.std(ddof=ddof)) if len(r) > ddof else float("nan")
    return {
        "total_return": float((1.0 + r).prod() - 1.0) if len(r) else float("nan"),
        "annualized_return": annualized_return(r, periods_per_year),
        "annualized_vol": std * math.sqrt(periods_per_year),
        "sharpe": float(excess.mean()) / ex_std * math.sqrt(periods_per_year)
        if ex_std and ex_std == ex_std
        else float("nan"),
        "max_drawdown": max_drawdown(r),
    }


# Skill families: (pattern over the task specification, API summary shown when it matches).
_FAMILIES: list[tuple[str, str]] = [
    (
        r"black.?scholes|\bgreeks?\b|implied vol|european (call|put|option)",
        "- bs_price(S, K, T, r, sigma, option_type='call', q=0.0); bs_greeks(...) -> dict delta,\n"
        "  gamma, vega (per 1.0 sigma), theta (PER YEAR), rho (per 1.0 r); implied_vol(price, S, K,\n"
        "  T, r, option_type, q). T in years, rates continuous, sigma decimal; arrays accepted.",
    ),
    (
        r"duration|convexity|yield to maturity|\bytm\b|coupon bond|bond pric",
        "- bond_price(face, coupon_rate, ytm, years, freq=2); bond_ytm(price, face, coupon_rate,\n"
        "  years, freq=2); bond_duration_convexity(...) -> dict price, macaulay_duration,\n"
        "  modified_duration, convexity. Annual rates compounded `freq` times a year.",
    ),
    (
        r"value.at.risk|\bvar\b|expected shortfall|\bcvar\b|\bes\b",
        "- historical_var_es(returns, alpha=0.99, method='linear') and parametric_var_es(mu, sigma,\n"
        "  alpha=0.99) -> dict var, es as POSITIVE losses (alpha is the confidence level).",
    ),
    (
        r"monte.?carlo|geometric brownian|\bgbm\b|simulat\w+ path",
        "- gbm_paths(S0, r, sigma, T, steps, n_paths, seed=None, q=0.0, antithetic=False) ->\n"
        "  array (n_paths, steps + 1), exact risk-neutral GBM from numpy default_rng(seed).",
    ),
    (
        r"sharpe|drawdown|annuali[sz]ed (return|vol)|cumulative return",
        "- performance_summary(returns, periods_per_year, risk_free_per_period=0.0, ddof=1) ->\n"
        "  dict total_return, annualized_return (geometric), annualized_vol, sharpe, max_drawdown\n"
        "  (negative); also annualized_return, annualized_vol, sharpe_ratio, max_drawdown(values,\n"
        "  is_returns=True), cumulative_returns.",
    ),
]

_SKILLS_HEADER = (
    "Helper module `agent_skills` (importable next to your script). Textbook reference\n"
    "implementations, tested against analytic values. Conventions are arguments: set them to\n"
    "what the TASK SPECIFICATION says, and if the specification defines a formula differently,\n"
    "follow the specification.\n"
    "- write_json(path, obj): REQUIRED for every JSON deliverable; strict JSON, NaN/inf -> null,\n"
    "  numpy/pandas values ok."
)


def skills_summary(instruction: str) -> str:
    """The prompt's skills section: the always-useful writer plus the families this task needs."""
    import re

    lines = [_SKILLS_HEADER]
    lines += [doc for pattern, doc in _FAMILIES if re.search(pattern, instruction, re.I)]
    return "\n".join(lines)
