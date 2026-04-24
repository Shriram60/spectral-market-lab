# Stock Price Trend Approximation using Linear Algebra

A Python project that applies a full linear algebra pipeline to approximate stock price trends using least squares regression. Fetches live closing prices via the `yfinance` API and solves the underlying overdetermined system rigorously — through rank-nullity analysis, Gram-Schmidt orthogonalization, projection onto the column space, Normal Equations, and eigenvalue decomposition.

---

## Overview

Given *n* days of a stock's closing prices, no single straight line passes exactly through all data points. This project frames trend fitting as an overdetermined linear system **Ax = b** and finds the mathematically optimal solution — the one that minimizes total squared error — using core linear algebra techniques.

The result is a trend line of the form:

```
Price(t) = c0 + c1 * t
```

where `c0` is the intercept and `c1` is the slope (price change per day).

---

## Linear Algebra Pipeline

### 1. Design Matrix Construction
The *n* data points are expressed as a matrix equation **Ax = b**, where **A** is an *n×2* design matrix (first column of ones, second column of day indices), **x = [c0, c1]** is the unknown coefficient vector, and **b** is the vector of actual closing prices.

### 2. Rank and Nullity Analysis
`numpy.linalg.matrix_rank(A)` confirms that rank(**A**) = 2 and nullity = 0, meaning:
- Both columns are linearly independent
- The system has a unique least squares solution
- **A^T A** is invertible (guaranteed by the Rank-Nullity theorem)

### 3. Gram-Schmidt Orthogonalization (QR Decomposition)
The two column vectors of **A** are linearly independent but not orthogonal. Gram-Schmidt converts them into an orthonormal basis spanning the same subspace. In code this is done via `numpy.linalg.qr(A)`, which computes the QR decomposition directly.

### 4. Projection onto the Column Space
The price vector **b** lives in ℝⁿ but the column space of **A** is only a 2-dimensional plane inside that space. Since **b** does not lie exactly on this plane, the best approximation **b̂** is its orthogonal projection onto C(**A**):

```
b̂ = A (AᵀA)⁻¹ Aᵀ b
```

The residual vector **e = b − b̂** is perpendicular to C(**A**), which is what makes this the minimum-error solution.

### 5. Least Squares via Normal Equations
Minimizing `||Ax − b||²` and setting the derivative to zero yields the Normal Equations:

```
AᵀA · x̂ = Aᵀb
```

Solved using `numpy.linalg.solve(ATA, ATb)` to obtain the intercept `c0` and slope `c1`.

### 6. Eigenvalue Analysis
Eigenvalues of the symmetric matrix **AᵀA** reveal the dominant patterns in the data. A large eigenvalue ratio (dominant / minor) confirms that a single linear trend explains the vast majority of price variance, with the minor eigenvalue corresponding to daily noise.

---

## Results (Muthoot Finance — MUTHOOTFIN.NS)

```
Trend line:  Price(t) = 3480.54 + 14.07 * t
Slope:       +₹14.07 per day  →  clear upward trend

Eigenvalues of AᵀA:  [210.40,  1.60]
Eigenvalue ratio:     131.75×
Variance explained:  ~98% driven by a single linear trend

Day 9  predicted price:  ₹3607.20
Day 10 predicted price:  ₹3621.27
```

| Day | Actual (₹) | Predicted (₹) | Residual (₹) |
|-----|-----------|--------------|-------------|
| 1   | 3446.06   | 3494.62      | −48.55      |
| 2   | 3543.93   | 3508.69      | +35.24      |
| 3   | 3532.43   | 3522.76      | +9.67       |
| 4   | 3567.83   | 3536.83      | +30.99      |
| 5   | 3548.00   | 3550.91      | −2.91       |
| 6   | 3571.30   | 3564.98      | +6.32       |
| 7   | 3531.20   | 3579.05      | −47.85      |
| 8   | 3610.20   | 3593.12      | +17.08      |

---

## Output

The script generates a two-panel plot saved as `muthoot_trend.png`:

- **Left panel** — Actual closing prices, fitted trend line, and 2-day forecast (starred markers)
- **Right panel** — Residual bar chart showing per-day prediction error

---

## Project Structure

```
Stock-Trend-Approximation/
├── Stock_Prediction.py   # Full linear algebra pipeline
├── muthoot_trend.png     # Output plot (generated on run)
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.7+

### Installation

```bash
# Clone the repository
git clone https://github.com/Shriram60/Stock-Trend-Approximation.git
cd Stock-Trend-Approximation

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install numpy matplotlib yfinance
```

### Run

```bash
python Stock_Prediction.py
```

The script fetches the last 8 trading days of live data automatically — no manual data entry required.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Matrix operations, QR decomposition, eigenvalue analysis |
| `matplotlib` | Trend line and residual plots |
| `yfinance` | Live stock price data via Yahoo Finance API |

---

## Limitations

- Linear model only — does not capture non-linear price movements
- Based on 8 data points; predictions are reliable only for short-range extrapolation
- Does not account for volatility, news events, or macroeconomic factors
- Live data means results change each run as new trading days are fetched

## Possible Extensions

- Polynomial fitting (add t² column) for quadratic trend approximation
- SVD-based solution instead of Normal Equations for better numerical stability
- Multi-feature regression incorporating volume or open price as additional columns
- Rolling window analysis to track how the trend line evolves over time
---

*Mini project submitted for Linear Algebra and Its Applications — PES University, 2024–25.*
