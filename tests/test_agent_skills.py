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


@pytest.mark.parametrize("df, loc, scale, alpha", [(3, 0, 1, 0.95), (8, -0.7, 1.8, 0.99)])
def test_student_t_risk_matches_independent_tail_integration(df, loc, scale, alpha):
    from scipy.integrate import quad
    from scipy.stats import t

    result = skills.student_t_var_es(df, loc, scale, alpha)
    distribution = t(df, loc=loc, scale=scale)
    assert distribution.cdf(result["var"]) == pytest.approx(alpha, abs=1e-12)
    integral = quad(lambda x: x * distribution.pdf(x), result["var"], np.inf)[0]
    assert result["es"] == pytest.approx(integral / (1 - alpha), rel=1e-8)


def test_normal_mixture_reduces_to_single_normal_and_respects_loss_convention():
    from scipy.stats import norm

    mean, sigma, alpha = -0.4, 1.7, 0.975
    expected_var = mean + sigma * norm.ppf(alpha)
    expected_es = mean + sigma * norm.pdf(norm.ppf(alpha)) / (1 - alpha)
    result = skills.normal_mixture_var_es([3, 7], [mean, mean], [sigma, sigma], alpha)
    assert result["var"] == pytest.approx(expected_var, abs=1e-10)
    assert result["es"] == pytest.approx(expected_es, abs=1e-10)
    # The existing Normal helper takes RETURNS; the new ones take LOSSES.
    assert result == pytest.approx(skills.parametric_var_es(-mean, sigma, alpha))
    assert skills.normal_mixture_var_es(
        [1e308, 1e308], [mean, mean], [sigma, sigma], alpha
    ) == pytest.approx(result)


def test_heterogeneous_mixture_tail_matches_density_integration():
    from scipy.integrate import quad
    from scipy.stats import norm

    weights, means, sigmas, alpha = np.array([0.7, 0.3]), [-0.3, 1.2], [0.8, 2.0], 0.97
    result = skills.normal_mixture_var_es(weights, means, sigmas, alpha)

    def density(x):
        return float(weights @ norm.pdf(x, loc=means, scale=sigmas))

    cdf = quad(density, -np.inf, result["var"])[0]
    tail = quad(lambda x: x * density(x), result["var"], np.inf)[0]
    assert cdf == pytest.approx(alpha, abs=1e-10)
    assert result["es"] == pytest.approx(tail / (1 - alpha), rel=1e-9)


def test_kde_risk_matches_scipy_density_and_cdf_integrals():
    from scipy.integrate import quad
    from scipy.stats import gaussian_kde

    losses = np.array([-2.0, -0.4, 0.2, 0.8, 2.1])
    weights, alpha = [1, 2, 1, 3, 1], 0.95
    fitted = gaussian_kde(losses, bw_method="silverman", weights=weights)
    result = skills.kde_var_es(losses, alpha, weights=weights)
    assert fitted.integrate_box_1d(-np.inf, result["var"]) == pytest.approx(alpha, abs=1e-12)
    tail = quad(lambda x: x * float(fitted([x])[0]), result["var"], np.inf)[0]
    assert result["es"] == pytest.approx(tail / (1 - alpha), rel=1e-9)


def test_kde_grid_interpolation_and_location_scale_conventions():
    losses = np.array([-2.0, -0.4, 0.2, 0.8, 2.1])
    direct = skills.kde_var_es(losses, 0.975)
    grid = skills.kde_var_es(losses, 0.975, quantile_grid_size=1024)
    assert grid == pytest.approx(direct, rel=1e-4)
    transformed = skills.kde_var_es(2 * losses + 3, 0.975)
    assert transformed == pytest.approx({key: 2 * value + 3 for key, value in direct.items()})


@pytest.mark.parametrize("df, scale, alpha", [(1, 1, 0.99), (3, 0, 0.99), (3, 1, 1.0)])
def test_undefined_student_t_tail_expectations_are_rejected(df, scale, alpha):
    with pytest.raises(ValueError):
        skills.student_t_var_es(df, scale=scale, alpha=alpha)


@pytest.mark.parametrize(
    "weights, means, sigmas",
    [([0, 0], [0, 1], [1, 1]), ([1, -1], [0, 1], [1, 1]), ([1], [0, 1], [1]), ([1], [0], [0])],
)
def test_invalid_mixture_parameters_are_rejected(weights, means, sigmas):
    with pytest.raises(ValueError):
        skills.normal_mixture_var_es(weights, means, sigmas)


def test_tail_skills_are_advertised_only_for_relevant_tasks():
    summary = skills.skills_summary("Fit a Student-t distribution and kernel density estimate.")
    for name in ("student_t_var_es", "normal_mixture_var_es", "kde_var_es", "quantile_grid_size"):
        assert name in summary
    assert "kde_var_es" not in skills.skills_summary("Compute a coupon bond's duration.")
