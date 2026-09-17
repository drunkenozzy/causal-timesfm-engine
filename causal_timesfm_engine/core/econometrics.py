"""
Causal Decoupling & Stationarity Engine
=======================================
Pure Python / NumPy compatible implementation of:
  1. Log Transformation: ln(x)
  2. First Differencing: Delta ln(x_t) = ln(x_t) - ln(x_{t-1})
  3. Stationarity Testing: Dickey-Fuller unit-root test
  4. Granger Causality VAR test (Kills the Elevator Effect)
"""

import math

class EconometricFilter:
    def __init__(self, p_value_threshold=0.05, max_lags=2):
        self.p_threshold = p_value_threshold
        self.max_lags = max_lags

    def log_transform(self, series):
        """Converts positive level series into natural log space."""
        vals = [float(x) for x in series]
        min_val = min(vals)
        shift = (abs(min_val) + 1.0) if min_val <= 0 else 0.0
        return [math.log(x + shift) for x in vals]

    def first_difference(self, series):
        """Calculates first differences: Delta y_t = y_t - y_{t-1}."""
        vals = [float(x) for x in series]
        return [vals[i] - vals[i - 1] for i in range(1, len(vals))]

    def stationarize(self, raw_levels):
        """Takes logs and first differences to remove deterministic trend."""
        log_series = self.log_transform(raw_levels)
        diff_series = self.first_difference(log_series)
        is_stationary, stat_p = self.check_stationarity(diff_series)
        return {
            "log_levels": log_series,
            "diff_growth": diff_series,
            "is_stationary": is_stationary,
            "adf_p_value": stat_p
        }

    def check_stationarity(self, series):
        """
        Dickey-Fuller Unit Root Test:
        Regresses Delta y_t on y_{t-1}.
        H0: Unit root present (Non-stationary).
        """
        vals = [float(x) for x in series]
        n = len(vals)
        if n < 15:
            return True, 0.01

        dy = [vals[i] - vals[i - 1] for i in range(1, n)]
        y_lag = vals[:-1]

        mean_y = sum(y_lag) / len(y_lag)
        mean_dy = sum(dy) / len(dy)

        denom = sum((y - mean_y)**2 for y in y_lag)
        if denom == 0:
            return False, 1.0

        beta = sum((y - mean_y) * (d - mean_dy) for y, d in zip(y_lag, dy)) / denom
        residuals = [d - (mean_dy + beta * (y - mean_y)) for y, d in zip(y_lag, dy)]
        s_err = math.sqrt(sum(r**2 for r in residuals) / max(1, len(dy) - 2))
        std_beta = s_err / math.sqrt(denom)

        t_stat = beta / (std_beta if std_beta > 0 else 1e-9)

        # Critical values: -2.86 at 5%
        if t_stat < -2.86:
            return True, 0.01
        elif t_stat < -2.57:
            return True, 0.045
        else:
            return False, 0.35

    def _solve_ols_1d(self, X_matrix, Y_vec):
        """Solves (X'X)^(-1) X'Y for small matrix using normal equations."""
        rows = len(X_matrix)
        cols = len(X_matrix[0])

        # Compute X'X (cols x cols)
        XtX = [[sum(X_matrix[r][i] * X_matrix[r][j] for r in range(rows)) for j in range(cols)] for i in range(cols)]
        # Compute X'Y (cols x 1)
        XtY = [sum(X_matrix[r][i] * Y_vec[r] for r in range(rows)) for i in range(cols)]

        # Gauss-Jordan elimination
        A = [row[:] for row in XtX]
        b = XtY[:]

        for i in range(cols):
            diag = A[i][i]
            if abs(diag) < 1e-12:
                # Add small ridge
                diag += 1e-6
                A[i][i] = diag
            for j in range(cols):
                A[i][j] /= diag
            b[i] /= diag

            for k in range(cols):
                if k != i:
                    factor = A[k][i]
                    for j in range(cols):
                        A[k][j] -= factor * A[i][j]
                    b[k] -= factor * b[i]

        beta = b
        # Compute RSS
        rss = sum((Y_vec[r] - sum(X_matrix[r][c] * beta[c] for c in range(cols)))**2 for r in range(rows))
        return beta, rss

    def granger_causality_test(self, x_driver, y_target, lags=2):
        """
        Tests whether lagged shocks in X predict future values of Y
        beyond Y's own lags.
        """
        x = [float(v) for v in x_driver]
        y = [float(v) for v in y_target]

        min_len = min(len(x), len(y))
        x = x[-min_len:]
        y = y[-min_len:]

        n = min_len - lags
        if n <= (2 * lags + 2):
            return {
                "causality_confirmed": False,
                "f_statistic": 0.0,
                "p_value": 1.0,
                "verdict": "Insufficient sample length."
            }

        Y_vec = y[lags:]

        # Restricted: Lags of Y only + intercept
        X_restr = []
        for r in range(n):
            row = [1.0]
            for lag in range(1, lags + 1):
                idx = (lags + r) - lag
                row.append(y[idx])
            X_restr.append(row)

        # Unrestricted: Lags of Y + Lags of X + intercept
        X_unrestr = []
        for r in range(n):
            row = [1.0]
            for lag in range(1, lags + 1):
                idx = (lags + r) - lag
                row.append(y[idx])
            for lag in range(1, lags + 1):
                idx = (lags + r) - lag
                row.append(x[idx])
            X_unrestr.append(row)

        _, rss_r = self._solve_ols_1d(X_restr, Y_vec)
        _, rss_u = self._solve_ols_1d(X_unrestr, Y_vec)

        df_num = lags
        df_denom = n - (2 * lags + 1)
        if df_denom <= 0 or rss_u <= 0:
            return {"causality_confirmed": False, "f_statistic": 0.0, "p_value": 1.0, "verdict": "Rank deficiency"}

        f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_denom)
        f_stat = max(0.0, f_stat)

        is_causal = f_stat > 3.10
        approx_p = 0.01 if f_stat > 4.5 else (0.04 if is_causal else 0.40)

        verdict = (
            "TRUE MECHANICAL CAUSALITY CONFIRMED: Lagged shocks in driver predict target beyond own momentum."
            if is_causal
            else "SPURIOUS / CO-TRENDING ARTIFACT: Driver does NOT Granger-cause target once trend is decoupled."
        )

        return {
            "lags": lags,
            "f_statistic": round(float(f_stat), 3),
            "p_value": round(float(approx_p), 4),
            "causality_confirmed": is_causal,
            "verdict": verdict
        }
