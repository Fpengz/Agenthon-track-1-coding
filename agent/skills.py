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
solution's misconceptions. Analytic tests live in tests/test_base_agent.py and
tests/test_agent_skills.py.
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


def _return_conventions(return_type: str, annualization: str = "geometric") -> None:
    if return_type not in {"simple", "log"}:
        raise ValueError("return_type must be 'simple' or 'log' (decimal returns)")
    if annualization not in {"geometric", "arithmetic"}:
        raise ValueError("annualization must be 'geometric' or 'arithmetic'")


def annualized_return(
    returns: Any,
    periods_per_year: int,
    *,
    return_type: str = "simple",
    annualization: str = "geometric",
) -> float:
    """Annualised decimal return under an explicit convention.

    geometric: simple returns compound via prod(1+r); log returns via exp(sum(r)),
    then annualise the wealth growth. arithmetic: mean(r)*periods_per_year; for log
    returns this is an annual LOG return, without exponentiation.
    """
    _return_conventions(return_type, annualization)
    r = _series(returns)
    if r.empty:
        return float("nan")
    if annualization == "arithmetic":
        return float(r.mean()) * periods_per_year
    if return_type == "log":
        return math.expm1(float(r.mean()) * periods_per_year)
    growth = float((1.0 + r).prod())
    return growth ** (periods_per_year / len(r)) - 1.0


def annualized_vol(returns: Any, periods_per_year: int, *, ddof: int = 1) -> float:
    """Standard deviation of periodic returns scaled by sqrt(periods); ddof=1 is sample,
    ddof=0 population. Pass the representation required by the specification unchanged."""
    r = _series(returns)
    if len(r) <= ddof:
        return float("nan")
    return float(r.std(ddof=ddof)) * math.sqrt(periods_per_year)


def sharpe_ratio(
    returns: Any,
    periods_per_year: int,
    risk_free_per_period: float = 0.0,
    *,
    ddof: int = 1,
    return_type: str = "simple",
    annualization: str = "geometric",
    sharpe_method: str = "arithmetic",
    risk_free_annual: float | None = None,
) -> float:
    """Annualised excess return divided by annualised volatility.

    arithmetic uses mean(r)*periods (an annual log mean for log returns);
    annualized_return uses annualized_return(..., annualization=...) as its numerator.
    Subtract risk_free_annual directly, or risk_free_per_period*periods if not supplied.
    Never supply both risk-free arguments with a nonzero per-period rate. The specification
    decides whether arithmetic Sharpe or CAGR-based Sharpe is wanted.
    """
    _return_conventions(return_type, annualization)
    if sharpe_method not in {"arithmetic", "annualized_return"}:
        raise ValueError("sharpe_method must be 'arithmetic' or 'annualized_return'")
    if risk_free_annual is not None and risk_free_per_period != 0.0:
        raise ValueError("supply either risk_free_annual or risk_free_per_period, not both")
    r = _series(returns)
    if len(r) <= ddof:
        return float("nan")
    # Preserve the existing per-period excess-return calculation for default callers.
    excess = r - risk_free_per_period
    std = float(excess.std(ddof=ddof))
    if std == 0.0:
        return float("nan")
    if sharpe_method == "arithmetic" and risk_free_annual is None:
        return float(excess.mean()) / std * math.sqrt(periods_per_year)
    annual_return = (
        float(r.mean()) * periods_per_year
        if sharpe_method == "arithmetic"
        else annualized_return(
            r, periods_per_year, return_type=return_type, annualization=annualization
        )
    )
    annual_rf = (
        risk_free_annual
        if risk_free_annual is not None
        else risk_free_per_period * periods_per_year
    )
    return (annual_return - annual_rf) / (std * math.sqrt(periods_per_year))


def cumulative_returns(returns: Any, *, return_type: str = "simple") -> pd.Series:
    """Compounded cumulative return path; simple: prod(1+r)-1, log: exp(cumsum(r))-1.
    Missing observations are treated as zero, preserving the supplied index."""
    _return_conventions(return_type)
    r = pd.Series(returns, dtype=float).fillna(0.0)
    if return_type == "log":
        return pd.Series(np.expm1(r.cumsum().to_numpy()), index=r.index, name=r.name)
    return (1.0 + r).cumprod() - 1.0


def max_drawdown(
    values: Any,
    is_returns: bool = True,
    *,
    return_type: str = "simple",
    include_initial: bool = True,
) -> float:
    """Maximum drawdown as a NEGATIVE fraction, from returns or an equity curve.
    For returns, include_initial=True includes starting wealth 1.0; False uses only the
    observed compounded path. Equity curves are used as supplied, without prepending wealth.
    """
    _return_conventions(return_type)
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return float("nan")
    if is_returns:
        equity = (
            pd.Series(np.exp(series.cumsum().to_numpy()), index=series.index)
            if return_type == "log"
            else (1.0 + series).cumprod()
        )
    else:
        equity = series
    if is_returns and include_initial:
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


def student_t_var_es(
    df: float, loc: float = 0.0, scale: float = 1.0, alpha: float = 0.99
) -> dict[str, float]:
    """Upper-tail risk for Student-t LOSSES, with scipy's location-scale convention.

    ES exists for df > 1. No variance standardisation is applied. For return data, use
    losses = -returns and negate the fitted location; keep scale unchanged.
    """
    from scipy.stats import t

    if not (df > 1 and scale > 0 and 0 < alpha < 1):
        raise ValueError("Student-t ES needs df > 1, scale > 0 and 0 < alpha < 1")
    if not all(math.isfinite(v) for v in (df, loc, scale)):
        raise ValueError("Student-t parameters must be finite")
    z = float(t.ppf(alpha, df))
    tail_mean = (df + z * z) / (df - 1) * float(t.pdf(z, df)) / (1 - alpha)
    return {"var": loc + scale * z, "es": loc + scale * tail_mean}


def normal_mixture_var_es(
    weights: Any,
    means: Any,
    sigmas: Any,
    alpha: float = 0.99,
    *,
    quantile_grid_size: int | None = None,
) -> dict[str, float]:
    """Upper-tail VaR/ES for a mixture of normal LOSS distributions.

    Component sigmas are standard deviations, not variances. Weights are normalised. VaR
    inverts the mixture CDF; ES uses exact Gaussian tail first moments. Set quantile_grid_size
    for a fine-grid CDF interpolation when the specification requires that procedure.
    """
    from scipy.optimize import brentq
    from scipy.stats import norm

    w, mu, sd = (np.asarray(v, dtype=float) for v in (weights, means, sigmas))
    if not (
        0 < alpha < 1
        and w.ndim == mu.ndim == sd.ndim == 1
        and w.size == mu.size == sd.size
        and w.size > 0
        and all(np.isfinite(v).all() for v in (w, mu, sd))
        and (w >= 0).all()
        and (w > 0).any()
        and (sd > 0).all()
    ):
        raise ValueError("Need finite mixture vectors, nonnegative weights, positive sigmas/alpha")
    w = w / w.max()  # avoid overflow when unnormalised weights are large
    w = w / w.sum()
    component_quantiles = mu + sd * norm.ppf(alpha)
    low = float(component_quantiles.min() - sd.max())
    high = float(component_quantiles.max() + sd.max())

    def gap(x: float) -> float:
        z = (x - mu) / sd
        # Survival probabilities avoid cancellation for high confidence levels.
        if alpha > 0.5:
            return float((1 - alpha) - w @ norm.sf(z))
        return float(w @ norm.cdf(z) - alpha)

    if quantile_grid_size is None:
        value = float(brentq(gap, low, high, xtol=1e-12))
    else:
        if quantile_grid_size < 2:
            raise ValueError("quantile_grid_size must be at least 2")
        grid = np.linspace(low, high, quantile_grid_size)
        # Bound memory even when a KDE has thousands of mixture components.
        cdf = np.concatenate(
            [
                norm.cdf((chunk[:, None] - mu) / sd) @ w
                for chunk in np.array_split(grid, max(1, (len(grid) + 127) // 128))
            ]
        )
        value = float(np.interp(alpha, cdf, grid))
    z = (value - mu) / sd
    es = float(w @ (mu * norm.sf(z) + sd * norm.pdf(z)) / (1 - alpha))
    return {"var": value, "es": es}


def kde_var_es(
    losses: Any,
    alpha: float = 0.99,
    bw_method: Any = "silverman",
    *,
    weights: Any = None,
    quantile_grid_size: int | None = None,
) -> dict[str, float]:
    """One-dimensional Gaussian KDE risk for LOSSES, using scipy bandwidth conventions.

    A Gaussian KDE is a normal mixture: its CDF and ES can be computed without repeated
    numerical integration of the density. Pass quantile_grid_size for CDF grid interpolation,
    or leave it None for direct CDF inversion. Convert returns to losses before calling.
    """
    from scipy.stats import gaussian_kde

    fitted = gaussian_kde(np.asarray(losses, dtype=float), bw_method=bw_method, weights=weights)
    values = fitted.dataset[0]
    sigma = math.sqrt(float(fitted.covariance[0, 0]))
    return normal_mixture_var_es(
        fitted.weights,
        values,
        np.full(values.shape, sigma),
        alpha,
        quantile_grid_size=quantile_grid_size,
    )


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
    returns: Any,
    periods_per_year: int,
    risk_free_per_period: float = 0.0,
    ddof: int = 1,
    *,
    return_type: str = "simple",
    annualization: str = "geometric",
    sharpe_method: str = "arithmetic",
    risk_free_annual: float | None = None,
    include_initial: bool = True,
) -> dict[str, float]:
    """Common performance statistics with explicit return/annualisation/Sharpe conventions.

    Defaults preserve SIMPLE returns, geometric annualised return, arithmetic Sharpe,
    sample std (ddof=1), and negative drawdown including starting wealth 1.0. Log returns
    compound via exp(cumsum(r)); arithmetic annualisation reports the annual log mean.
    sharpe_method='annualized_return' uses the chosen annualised return rather than the
    arithmetic mean. Match every argument to the specification rather than relying on defaults.
    """
    _return_conventions(return_type, annualization)
    r = _series(returns)
    total = math.expm1(float(r.sum())) if return_type == "log" else float((1.0 + r).prod() - 1.0)
    return {
        "total_return": total if len(r) else float("nan"),
        "annualized_return": annualized_return(
            r, periods_per_year, return_type=return_type, annualization=annualization
        ),
        "annualized_vol": annualized_vol(r, periods_per_year, ddof=ddof),
        "sharpe": sharpe_ratio(
            r,
            periods_per_year,
            risk_free_per_period,
            ddof=ddof,
            return_type=return_type,
            annualization=annualization,
            sharpe_method=sharpe_method,
            risk_free_annual=risk_free_annual,
        ),
        "max_drawdown": max_drawdown(r, return_type=return_type, include_initial=include_initial),
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
        r"student.?t|heavy.tail|gaussian.{0,30}mixture|contaminated normal|kernel density|\bkde\b",
        "- student_t_var_es(df, loc=0.0, scale=1.0, alpha=0.99): scipy location-scale t, df>1.\n"
        "- normal_mixture_var_es(weights, means, sigmas, alpha=0.99): sigmas are standard\n"
        "  deviations; exact mixture CDF inversion and analytic tail first moment.\n"
        "- kde_var_es(losses, alpha=0.99, bw_method='silverman', weights=None): Gaussian KDE\n"
        "  with exact CDF inversion and tail moments, avoiding costly density integration.\n"
        "  Both mixture and KDE accept quantile_grid_size=N for CDF grid interpolation.\n"
        "  All three return dict var, es for the UPPER TAIL of LOSSES; convert returns to\n"
        "  losses by negating them, including the fitted location for Student-t.",
    ),
    (
        r"monte.?carlo|geometric brownian|\bgbm\b|simulat\w+ path",
        "- gbm_paths(S0, r, sigma, T, steps, n_paths, seed=None, q=0.0, antithetic=False) ->\n"
        "  array (n_paths, steps + 1), exact risk-neutral GBM from numpy default_rng(seed).",
    ),
    (
        r"sharpe|drawdown|annuali[sz]ed (return|vol)|cumulative return",
        "- performance_summary(returns, periods_per_year, risk_free_per_period=0.0, ddof=1,\n"
        "  return_type='simple', annualization='geometric', sharpe_method='arithmetic',\n"
        "  risk_free_annual=None, include_initial=True) -> dict total_return, annualized_return,\n"
        "  annualized_vol, sharpe, max_drawdown (negative). CHOOSE conventions from the spec:\n"
        "  return_type='log' compounds exp(sum(r)); annualization='arithmetic' means mean(r)*\n"
        "  periods (annual LOG mean for log returns), 'geometric' means compounded CAGR;\n"
        "  sharpe_method='annualized_return' uses the selected annualised return in Sharpe,\n"
        "  default 'arithmetic' uses mean(r)*periods; ddof=0 is population, 1 sample. Supply\n"
        "  risk_free_annual OR risk_free_per_period. include_initial=False excludes wealth 1.0\n"
        "  from drawdown. Also annualized_return, annualized_vol, sharpe_ratio, max_drawdown,\n"
        "  cumulative_returns with corresponding keywords. Do not apply simple-return\n"
        "  compounding to log returns or substitute CAGR for a required arithmetic mean.",
    ),
]

_SKILLS_HEADER = (
    "Helper module `agent_skills` (importable next to your script). Textbook reference\n"
    "implementations, tested against analytic values. Conventions are arguments: set them to\n"
    "what the TASK SPECIFICATION says. Use the reference helpers for matching calculations\n"
    "rather than reimplementing their formulas. If the specification defines a formula differently,\n"
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
