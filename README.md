# Spectral Market Lab

Work in progress. The finished project will be a from-scratch numerical linear algebra library (QR, SVD, eigensolvers, ridge/LASSO, constrained QP) applied to 51 NSE large-cap stocks.
It will cover the spectral structure of the market: eigenportfolios, random-matrix-theory noise filtering and unsupervised sector discovery through spectral clustering.
On top of that come practical applications evaluated out of sample: sparse index tracking, minimum-variance portfolios with cleaned covariance estimates and PCA-residual statistical arbitrage.
It will also include time-varying trend and regime models (Kalman filter, hidden Markov model) as the successor to the original straight-line trend fit.
Every result will be reproducible offline from cached data, tested against NumPy/SciPy/scikit-learn and checked by CI.
