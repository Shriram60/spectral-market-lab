"""Run every experiment: ``python -m spectral_market.experiments``."""

from . import (
    clustering,
    index_tracking,
    kalman_trend,
    min_variance,
    orthogonality,
    regimes,
    spectrum,
    stability,
    stat_arb,
    trend_report,
)

MODULES = (
    orthogonality,
    stability,
    trend_report,
    spectrum,
    clustering,
    index_tracking,
    min_variance,
    stat_arb,
    kalman_trend,
    regimes,
)

for module in MODULES:
    print(f"\n=== {module.__name__.rsplit('.', 1)[-1]} ===")
    module.main()
