import numpy as np
from scipy import stats


def diebold_mariano_test(errors_a, errors_b, horizon: int = 1):
    """Diebold & Mariano 1995, J. Bus. Econ. Stat. 13(3), 253-263, with the
    Harvey, Leybourne & Newbold 1997 small-sample correction (Int. J.
    Forecast. 13(2), 281-291).

    Tests H0: the two forecasts have equal expected squared error.
    """
    loss_a = errors_a**2
    loss_b = errors_b**2
    d = loss_a - loss_b
    n = len(d)

    d_mean = d.mean()
    autocovariance = np.array([
        np.mean((d[:n - lag] - d_mean) * (d[lag:] - d_mean)) for lag in range(horizon)
    ])
    long_run_variance = autocovariance[0] + 2 * autocovariance[1:].sum()
    dm_stat = d_mean / np.sqrt(long_run_variance / n)

    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm_stat_corrected = dm_stat * correction

    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_corrected), df=n - 1))
    return float(dm_stat_corrected), float(p_value)
