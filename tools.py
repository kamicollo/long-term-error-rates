import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

from typing import TypedDict, Protocol


class HasPDF(Protocol):
    def pdf(self) -> float:
        pass


class errorRates(TypedDict):
    power: float
    typeS: float
    typeM: float


def calculate_rates(delta: float, sd: float, n: int) -> errorRates:
    """Calculates power, Type S and Type M errors.

    Args:
        delta (float): Effect size
        sd (float): Standard deviation of effect size
        n (int): Sample size (in each group)

    Returns:
        errorRates: Power, Type S and Type M error estimates.
    """
    lmbda = abs(delta) / np.sqrt(sd**2 / n + sd**2 / n)
    z = norm.ppf(1 - 0.05 / 2)

    neg = norm.cdf(-z - lmbda)
    diff = norm.cdf(z - lmbda)
    pos = norm.cdf(z + lmbda)
    inv_diff = norm.cdf(lmbda - z)

    return {
        "power": neg + 1 - diff,
        "typeS": neg / (neg + 1 - diff),
        "typeM": (
            norm.pdf(lmbda + z) + norm.pdf(lmbda - z) + lmbda * (pos + inv_diff - 1)
        )
        / (lmbda * (1 - pos + inv_diff)),
    }


def detection_rate(baseline: float, required_sample_size: int, density_func: HasPDF):
    """Estimates long-run detection rate.

    Args:
        baseline (float): Metric baseline
        required_sample_size (int): Sample size (per group)
        density_func (HasPDF): Callable that implements .pdf() method to return density.
    """

    def inner_func(x):
        power = calculate_rates(
            n=required_sample_size, delta=x, sd=np.sqrt(baseline * (1 - baseline))
        )["power"]
        return density_func.pdf(x) * np.nan_to_num(power, nan=0)

    result, _ = quad(inner_func, -np.inf, np.inf)
    return result


def detectable_win_rate(
    baseline: float, required_sample_size: int, density_func: HasPDF
):
    """Estimates long-run win detection rate.

    Args:
        baseline (float): Metric baseline
        required_sample_size (int): Sample size (per group)
        density_func (HasPDF): Callable that implements .pdf() method to return density.
    """

    def inner_func(x):
        power = calculate_rates(
            n=required_sample_size, delta=x, sd=np.sqrt(baseline * (1 - baseline))
        )["power"]
        return 0 if x <= 0 else density_func.pdf(x) * np.nan_to_num(power, nan=0)

    result, _ = quad(inner_func, -np.inf, np.inf)
    return result


def sign_error_rate(baseline: float, required_sample_size: int, density_func: HasPDF):
    """Estimates long-run sign error rate.

    Args:
        baseline (float): Metric baseline
        required_sample_size (int): Sample size (per group)
        density_func (HasPDF): Callable that implements .pdf() method to return density.
    """

    def inner_func(x):
        r = calculate_rates(
            n=required_sample_size, delta=x, sd=np.sqrt(baseline * (1 - baseline))
        )
        return density_func.pdf(x) * np.nan_to_num(r["power"], nan=0) * r["typeS"]

    result, _ = quad(inner_func, -np.inf, np.inf)
    return result


def exaggeration_rate(baseline: float, required_sample_size: int, density_func: HasPDF):
    """Estimates long-run exaggeration (absolute size) rate.

    Args:
        baseline (float): Metric baseline
        required_sample_size (int): Sample size (per group)
        density_func (HasPDF): Callable that implements .pdf() method to return density.
    """

    def inner_func(x):
        r = calculate_rates(
            n=required_sample_size, delta=x, sd=np.sqrt(baseline * (1 - baseline))
        )
        return (
            density_func.pdf(x)
            * np.nan_to_num(r["power"], nan=0)
            * (r["typeM"] - 1)
            * abs(x)
        )

    result, _ = quad(inner_func, -np.inf, np.inf)
    return result
