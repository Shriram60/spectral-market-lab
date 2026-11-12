"""Run every Phase 1 experiment: ``python -m spectral_market.experiments``."""

from . import orthogonality, stability, trend_report

for module in (orthogonality, stability, trend_report):
    print(f"\n=== {module.__name__.rsplit('.', 1)[-1]} ===")
    module.main()
