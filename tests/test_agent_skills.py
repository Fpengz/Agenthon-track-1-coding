"""Analytic tests for the conventions offered to generated finance scripts."""

import math

import numpy as np
import pytest

from agent import skills


def test_arithmetic_annualisation_and_cagr_sharpe_are_distinct():
    returns = [0.10, -0.05, 0.02]
    periods = 12
    annual_mean = sum(returns) / len(returns) * periods
    cagr = (1.10 * 0.95 * 1.02) ** (periods / len(returns)) - 1
    vol = np.std(returns, ddof=1) * math.sqrt(periods)

    arithmetic = skills.performance_summary(returns, periods, annualization="arithmetic")
    assert arithmetic["annualized_return"] == pytest.approx(annual_mean)
    assert arithmetic["sharpe"] == pytest.approx(annual_mean / vol)
    cagr_metrics = skills.performance_summary(returns, periods, sharpe_method="annualized_return")
    assert cagr_metrics["annualized_return"] == pytest.approx(cagr)
    assert cagr_metrics["sharpe"] == pytest.approx(cagr / vol)
    assert cagr_metrics["sharpe"] != pytest.approx(arithmetic["sharpe"])


def test_log_return_metrics_use_log_wealth_and_population_std():
    returns = np.array([-0.20, 0.10, -0.05])
    periods, risk_free = 12, 0.03
    ann_log = returns.mean() * periods
    vol = returns.std(ddof=0) * math.sqrt(periods)
    wealth = np.exp(np.cumsum(returns))
    drawdown = np.min(wealth / np.maximum.accumulate(wealth) - 1)

    result = skills.performance_summary(
        returns,
        periods,
        ddof=0,
        return_type="log",
        risk_free_annual=risk_free,
        include_initial=False,
    )
    assert result["total_return"] == pytest.approx(math.expm1(returns.sum()))
    assert result["annualized_return"] == pytest.approx(math.expm1(ann_log))
    assert result["annualized_vol"] == pytest.approx(vol)
    assert result["sharpe"] == pytest.approx((ann_log - risk_free) / vol)
    assert result["max_drawdown"] == pytest.approx(drawdown)
    assert skills.annualized_return(
        returns, periods, return_type="log", annualization="arithmetic"
    ) == pytest.approx(ann_log)
    assert skills.cumulative_returns(returns, return_type="log").to_numpy() == pytest.approx(
        wealth - 1
    )


def test_return_representations_agree_on_compounded_wealth():
    simple = np.array([0.1, -0.2, 0.05])
    log = np.log1p(simple)
    assert skills.annualized_return(simple, 12) == pytest.approx(
        skills.annualized_return(log, 12, return_type="log")
    )
    assert skills.max_drawdown(simple) == pytest.approx(skills.max_drawdown(log, return_type="log"))
    assert skills.max_drawdown([-0.1]) == pytest.approx(-0.1)
    assert skills.max_drawdown([-0.1], include_initial=False) == pytest.approx(0.0)


def test_existing_default_metrics_and_positional_arguments_are_preserved():
    returns = np.array([0.10, -0.20, 0.05])
    periods, risk_free = 12, 0.001
    result = skills.performance_summary(returns, periods, risk_free, 1)
    expected_vol = returns.std(ddof=1) * math.sqrt(periods)
    assert result["total_return"] == pytest.approx(1.1 * 0.8 * 1.05 - 1)
    assert result["annualized_return"] == pytest.approx((1.1 * 0.8 * 1.05) ** 4 - 1)
    assert result["annualized_vol"] == pytest.approx(expected_vol)
    assert result["sharpe"] == pytest.approx((returns.mean() - risk_free) * periods / expected_vol)
    assert result["max_drawdown"] == pytest.approx(-0.2)
    assert skills.sharpe_ratio(returns, periods, risk_free) == pytest.approx(result["sharpe"])
    assert skills.annualized_vol(returns, periods, ddof=0) == pytest.approx(
        returns.std(ddof=0) * math.sqrt(periods)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"return_type": "percent"},
        {"annualization": "guess"},
        {"sharpe_method": "guess"},
        {"risk_free_per_period": 0.01, "risk_free_annual": 0.12},
    ],
)
def test_ambiguous_or_unknown_conventions_are_rejected(kwargs):
    with pytest.raises(ValueError):
        skills.performance_summary([0.1, -0.2, 0.05], 12, **kwargs)


def test_skills_summary_advertises_metric_conventions():
    summary = skills.skills_summary("Compute log-return Sharpe and annualized return.")
    for term in (
        "return_type",
        "annualization",
        "sharpe_method",
        "ddof",
        "risk_free_annual",
        "include_initial",
    ):
        assert term in summary
