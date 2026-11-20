"""Run every experiment: ``python -m spectral_market.experiments``."""

from . import clustering, index_tracking, orthogonality, spectrum, stability, trend_report

for module in (orthogonality, stability, trend_report, spectrum, clustering, index_tracking):
    print(f"\n=== {module.__name__.rsplit('.', 1)[-1]} ===")
    module.main()
