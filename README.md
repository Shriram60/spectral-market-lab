# Spectral Market Lab

[![CI](https://github.com/Shriram60/spectral-market-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Shriram60/spectral-market-lab/actions/workflows/ci.yml)

Numerical linear algebra implemented from scratch in Python and applied to Indian equity
markets: matrix factorisations, least squares, regularisation and statistically rigorous
evaluation of trend models.

## Key results

| Experiment | Result |
|---|---|
| QR stability | At κ(A) = 10⁸, loss of orthogonality is 0.75 for classical Gram-Schmidt, 3×10⁻⁹ for modified Gram-Schmidt and 1×10⁻¹⁵ for Householder. |
| Solver accuracy | On raw time indices, the normal equations fall below 6 correct digits at polynomial degree 4 and Cholesky fails at degree 12. Rescaling time to [−1, 1] lowers κ(A) by over 10 orders of magnitude and every solver stays accurate through degree 14. |
| Short-window trend | 8-session linear fit: R² = 0.002, slope 95% CI [−9.35, +10.16] INR/day. |
| Long-window trend | 5-year log-linear fit: R² = 0.59, Durbin-Watson = 0.007. Correcting for autocorrelation widens the growth-rate interval 29×. |
| Forecasting | Across 210 walk-forward configurations (239,904 forecasts), no trend model beats a random walk. 0/210 are significantly better in a Diebold-Mariano test. |

## Results

### QR factorisation stability

![Loss of orthogonality vs condition number](figures/orthogonality.png)

Loss of orthogonality ‖I − QᵀQ‖₂ follows the theoretical rates: ε·κ² for classical
Gram-Schmidt, ε·κ for modified Gram-Schmidt and ε for Householder reflections.

### Least-squares solvers and conditioning

![Normal equations vs QR vs SVD](figures/stability.png)

The experiment fits polynomial trends with known coefficients over 250 trading days. The
normal equations operate on κ(AᵀA) = κ(A)² and degrade first. The SVD solver truncates
singular values below ε·σ_max, so its error stays bounded on rank-deficient problems.
Rescaling the time axis matters more than the choice of solver, and a Legendre basis keeps
κ(A) below 6 at every degree tested.

### Trend diagnostics

![Trend diagnostics](figures/trend_diagnostics.png)

| Quantity | Value |
|---|---|
| Eigenvalues of AᵀA, A = [1, t], t = 1..8 | 210.40, 1.60. These are fixed by the time index and independent of prices. |
| κ(A), raw / standardised time | 11.48 / 1.53 |
| R², 8 sessions | 0.002 |
| Slope, 95% CI | +0.40 INR/day, [−9.35, +10.16] |
| Day 9 forecast, 95% prediction interval | 3761, [3681, 3841] |
| 5-year growth, OLS 95% CI | 21.3%/yr, [20.2, 22.4] |
| 5-year growth, returns-based 95% CI | 28.4%/yr, [−1.1, 66.6] |

Residuals of the 5-year fit have Durbin-Watson 0.007, which indicates near-unit-root
autocorrelation. OLS standard errors are therefore invalid; the returns-based interval is
the consistent estimate.

### Walk-forward evaluation

![Walk-forward evaluation](figures/walk_forward.png)

Seven series (MUTHOOTFIN, NIFTY 50, RELIANCE, TCS, HDFCBANK, INFY, ITC; 2021–2025) are
evaluated with rolling windows of 8–250 days, polynomial degrees 0–2 and horizons of 1 and
5 days. Every configuration has a higher RMSE than the random walk (median ratio
1.2–7.9×), and directional accuracy is 40.1–54.9%.

Rolling least-squares forecasts are computed as a fixed linear filter: ŷ = wᵀy with
w = QR⁻ᵀaₕ, derived once from a Householder QR. This turns the backtest into a single
matrix-vector product.

## Components

All factorisations are implemented from scratch. NumPy is used only for array arithmetic.

| Module | Contents |
|---|---|
| `linalg/triangular.py` | Forward and back substitution |
| `linalg/cholesky.py` | Cholesky factorisation with positive-definiteness detection |
| `linalg/qr.py` | Classical and modified Gram-Schmidt; compact Householder QR |
| `linalg/svd.py` | One-sided Jacobi SVD with QR preconditioning and round-robin ordering |
| `linalg/lstsq.py` | Least squares via normal equations, QR and SVD (minimum-norm) |
| `linalg/regularized.py` | Ridge regression (SVD filter factors, GCV), LASSO (coordinate descent, KKT check) |
| `trend.py` | Polynomial trends: R², standard errors, leverage, prediction intervals, Durbin-Watson |
| `forecast.py` | Walk-forward evaluation, Diebold-Mariano test with HLN correction |

## Reproducing

```bash
git clone https://github.com/Shriram60/spectral-market-lab.git
cd spectral-market-lab
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
python -m spectral_market.experiments
```

Prices are fixed to 2021-01-01 – 2025-12-31 and cached in `data/`, so all results
reproduce offline. Each figure has a matching JSON file in `results/`. CI runs linting,
tests and the full experiment pipeline on Python 3.10 and 3.12.

## Limitations

- Results cover Indian large-cap equities over a single five-year period.
- Prices are split- and dividend-adjusted closes from Yahoo Finance.
- The Jacobi SVD targets matrices of up to a few hundred columns; it is not a substitute
  for LAPACK at scale.

---

Originally a course project for *Linear Algebra and Its Applications*, PES University (2024–25).
