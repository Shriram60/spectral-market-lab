import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# 1. FETCH REAL DATA (last 8 days)
stock = yf.download("MUTHOOTFIN.NS", period="10d", interval="1d")
prices = stock['Close'].dropna().values[-8:].flatten()
days = np.arange(1, len(prices) + 1, dtype=float)

# 2. DESIGN MATRIX
n = len(days)
A = np.column_stack([np.ones(n), days])
b = prices  # already 1D

# 3. RANK & NULLITY
rank = np.linalg.matrix_rank(A)
print("Rank of A:", rank)
print("Nullity of A:", A.shape[1] - rank)

# 4. QR DECOMPOSITION
Q, R = np.linalg.qr(A)
print("Orthonormal basis Q (col 1):", Q[:, 0].round(4))

# 5. NORMAL EQUATIONS
ATA = A.T @ A
ATb = A.T @ b
print("A^T A:", ATA)
print("A^T b:", ATb)

# 6. LEAST SQUARES
x_hat = np.linalg.solve(ATA, ATb)
c0, c1 = float(x_hat[0]), float(x_hat[1])  # explicit float conversion
print("Intercept c0 =", round(c0, 4))
print("Slope     c1 =", round(c1, 4))
print(f"Trend line: Price(t) = {c0:.2f} + {c1:.2f} * t")

# 7. PREDICTIONS
predicted = (A @ x_hat).flatten()
residuals = (b - predicted).flatten()
print("\nDay | Actual | Predicted | Residual")
for i in range(n):
    print(i+1, round(b[i], 2), round(predicted[i], 2), round(residuals[i], 4))

# 8. EIGEN ANALYSIS
evals, _ = np.linalg.eig(ATA)
evals = np.sort(evals)[::-1]
print("Eigenvalues of ATA:", evals.round(4))
print("Eigenvalue ratio:", round(evals[0] / evals[1], 2))

# 9. FUTURE PREDICTIONS
for day in [n + 1, n + 2]:
    print(f"Day {day} predicted price: ₹", round(c0 + c1 * day, 2))

# 10. PLOT
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Residual lines
for d, p, pr in zip(days, b, predicted):
    ax1.plot([d, d], [float(p), float(pr)], 'gray', linestyle='--', linewidth=0.8)

ax1.scatter(days, b, color='steelblue', zorder=3, label='Actual')
ax1.plot(days, predicted, color='tomato', label='Trend line')
future_days = np.array([n + 1, n + 2])
ax1.scatter(future_days, [c0 + c1 * d for d in future_days],
            color='green', marker='*', s=120, zorder=4, label='Forecast')
ax1.set_title('Muthoot Finance — Least Squares Trend')
ax1.set_xlabel('Day')
ax1.set_ylabel('Price (₹)')
ax1.legend()

ax2.bar(days, residuals, color='steelblue')
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_title('Residuals')
ax2.set_xlabel('Day')
ax2.set_ylabel('Residual (₹)')

plt.tight_layout()
plt.savefig('muthoot_trend.png', dpi=150)
plt.show()
