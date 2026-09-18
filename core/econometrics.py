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
        stat_res = self.check_stationarity(diff_series)
        is_stationary, stat_p = stat_res[0], stat_res[1]
        method_used = stat_res[2] if len(stat_res) > 2 else "Dickey-Fuller OLS"
        return {
            "log_levels": log_series,
            "diff_growth": diff_series,
            "is_stationary": is_stationary,
            "p_value": stat_p,
            "adf_p_value": stat_p,
            "method": method_used
        }

    def check_stationarity(self, series):
        """
        Dickey-Fuller / Augmented Dickey-Fuller Unit Root Test:
        Tests H0: Unit root present (Non-stationary).
        Uses statsmodels.tsa.stattools.adfuller if installed for exact MacKinnon p-values;
        falls back to pure-Python first-order Dickey-Fuller OLS when statsmodels is absent.
        """
        vals = [float(x) for x in series]
        n = len(vals)
        if n < 20:
            return False, 1.0, "INSUFFICIENT_SAMPLE (<20 observations)"

        # 1. Preferred Institutional Path: statsmodels ADF with AIC lag selection
        try:
            from statsmodels.tsa.stattools import adfuller
            res = adfuller(vals, autolag='AIC')
            t_stat = float(res[0])
            p_val = float(res[1])
            lags_used = int(res[2])
            is_stat = p_val < self.p_threshold
            return is_stat, round(p_val, 4), f"Statsmodels ADF (MacKinnon p-value, AIC lags={lags_used})"
        except ImportError:
            pass

        # 2. Transparent Pure-Python Fallback: First-Order Dickey-Fuller OLS
        dy = [vals[i] - vals[i - 1] for i in range(1, n)]
        y_lag = vals[:-1]

        mean_y = sum(y_lag) / len(y_lag)
        mean_dy = sum(dy) / len(dy)

        denom = sum((y - mean_y)**2 for y in y_lag)
        if denom == 0:
            return False, 1.0, "ZERO_VARIANCE_SERIES"

        beta = sum((y - mean_y) * (d - mean_dy) for y, d in zip(y_lag, dy)) / denom
        residuals = [d - (mean_dy + beta * (y - mean_y)) for y, d in zip(y_lag, dy)]
        s_err = math.sqrt(sum(r**2 for r in residuals) / max(1, len(dy) - 2))
        std_beta = s_err / math.sqrt(denom)

        t_stat = beta / (std_beta if std_beta > 0 else 1e-9)

        # Standard Dickey-Fuller asymptotic response surface approximation (with constant)
        # 1%: -3.43, 5%: -2.86, 10%: -2.57
        if t_stat <= -3.43:
            p_val = 0.01 * math.exp(t_stat + 3.43)
        elif t_stat <= -2.86:
            p_val = 0.01 + 0.04 * (t_stat - (-3.43)) / (-2.86 - (-3.43))
        elif t_stat <= -2.57:
            p_val = 0.05 + 0.05 * (t_stat - (-2.86)) / (-2.57 - (-2.86))
        else:
            p_val = min(1.0, 0.10 + 0.90 * (1.0 / (1.0 + math.exp(-1.5 * (t_stat + 2.57)))))

        is_stat = p_val < self.p_threshold
        method_desc = "First-Order Dickey-Fuller OLS (No lag augmentation; install statsmodels for automated AIC-lag ADF)"
        return is_stat, round(float(p_val), 4), method_desc

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

    def granger_causality_test(self, x_driver, y_target, lags=2, max_lag=None):
        """
        Tests whether lagged shocks in X predict future values of Y
        beyond Y's own lags.
        """
        if max_lag is not None:
            lags = max_lag
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
            return {"predictive_precedence": False, "f_statistic": 0.0, "p_value": 1.0, "evidence_grade": "RANK_DEFICIENT", "verdict": "Rank deficiency in regression matrix."}

        f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_denom)
        f_stat = max(0.0, f_stat)

        # Exact P-value from F-distribution via SciPy
        try:
            from scipy.stats import f as f_dist
            exact_p = float(f_dist.sf(f_stat, df_num, df_denom))
            p_calc_method = "SciPy exact F-distribution sf"
        except ImportError:
            # Heuristic exponential tail approximation when SciPy is absent
            exact_p = math.exp(-0.5 * f_stat) if f_stat > 0 else 1.0
            p_calc_method = "Heuristic exponential approximation (Install scipy for exact F p-value)"

        # Scientific Causal Taxonomy (Correlation != Causation)
        has_precedence = exact_p < self.p_threshold
        if exact_p < 0.01:
            grade = "ROBUST_PREDICTIVE_PRECEDENCE"
            verdict = "ROBUST PREDICTIVE PRECEDENCE: Lagged shocks in driver contain significant incremental predictive information beyond target's own lags. (Note: Establishes predictive precedence, NOT mechanical structural causality)."
        elif exact_p < 0.05:
            grade = "MODERATE_PREDICTIVE_PRECEDENCE"
            verdict = "MODERATE PREDICTIVE PRECEDENCE: Statistically significant lead-lag relationship detected at 5% level."
        elif exact_p < 0.10:
            grade = "WEAK_LEAD_LAG_EVIDENCE"
            verdict = "WEAK LEAD-LAG EVIDENCE: Marginal statistical precedence observed (0.05 < p <= 0.10). Inconclusive."
        else:
            grade = "NO_PREDICTIVE_EVIDENCE"
            verdict = "NO PREDICTIVE EVIDENCE: Driver does not Granger-predict target once trend and target's own lags are accounted for."

        return {
            "lags": lags,
            "f_statistic": round(float(f_stat), 3),
            "p_value": round(float(exact_p), 4),
            "exact_p_value": round(float(exact_p), 4),
            "p_value_method": p_calc_method,
            "predictive_precedence": has_precedence,
            "evidence_grade": grade,
            "epistemic_status": grade,
            "verdict": verdict
        }

    def compute_theils_u(self, actual, predicted):
        """
        Computes Theil's U Statistic (U-ratio against naive random walk):
          U = RMSE(forecast) / RMSE(naive random walk where y_hat_t = y_{t-1})
        Interpretation:
          U < 1.0: Model outperforms naive random walk (Hurdle PASSED).
          U = 1.0: Model equals naive persistence.
          U > 1.0: Model performs worse than naive guess (Hurdle FAILED).
        """
        if len(actual) != len(predicted) or len(actual) < 2:
            raise ValueError("Actual and predicted series must have matching length >= 2.")
        
        n = len(actual)
        forecast_mse = sum((predicted[i] - actual[i]) ** 2 for i in range(1, n)) / (n - 1)
        naive_mse = sum((actual[i - 1] - actual[i]) ** 2 for i in range(1, n)) / (n - 1)
        
        if naive_mse <= 1e-12:
            return 1.0 if forecast_mse <= 1e-12 else 999.0
            
        u = math.sqrt(forecast_mse) / math.sqrt(naive_mse)
        return round(float(u), 4)

    def evaluate_rolling_origin_theils_u(self, series, forecast_fn, min_train_len=30, horizon=1):
        """
        Executes an institutional rolling-origin (walk-forward) backtest evaluation:
        At each time step t from min_train_len to N - horizon:
          - Train history: series[:t]
          - Forecast y_hat_{t+horizon} using forecast_fn(train_history, horizon)
          - Compare against actual target series[t + horizon - 1]
          - Compare against naive random walk benchmark series[t - 1]
        
        Returns:
          {
            "theils_u": float,
            "hurdle_passed": bool,
            "n_evaluations": int,
            "forecast_rmse": float,
            "naive_rmse": float,
            "horizon": int
          }
        """
        series = [float(x) for x in series]
        n = len(series)
        if n < min_train_len + horizon:
            raise ValueError(f"Series length ({n}) insufficient for rolling-origin evaluation with min_train_len={min_train_len} and horizon={horizon}.")

        forecast_errors_sq = []
        naive_errors_sq = []

        for t in range(min_train_len, n - horizon + 1):
            train_history = series[:t]
            actual_val = series[t + horizon - 1]
            naive_pred = series[t - 1]

            pred_val = forecast_fn(train_history, horizon)
            if isinstance(pred_val, dict):
                pred_val = pred_val.get("p50_expected", pred_val.get("expected_target", pred_val.get("reconciled_p50", train_history[-1])))

            forecast_errors_sq.append((float(pred_val) - actual_val) ** 2)
            naive_errors_sq.append((naive_pred - actual_val) ** 2)

        m_fc = sum(forecast_errors_sq) / len(forecast_errors_sq)
        m_nv = sum(naive_errors_sq) / len(naive_errors_sq)

        rmse_fc = math.sqrt(m_fc)
        rmse_nv = math.sqrt(m_nv)

        if rmse_nv <= 1e-12:
            theils_u = 1.0 if rmse_fc <= 1e-12 else 999.0
        else:
            theils_u = rmse_fc / rmse_nv

        theils_u = round(float(theils_u), 4)
        hurdle_passed = bool(theils_u < 1.0)

        return {
            "theils_u": theils_u,
            "hurdle_passed": hurdle_passed,
            "n_evaluations": len(forecast_errors_sq),
            "forecast_rmse": round(rmse_fc, 4),
            "naive_rmse": round(rmse_nv, 4),
            "horizon": horizon
        }

    def evaluate_multi_horizon_theils_u(self, series, forecast_fn, min_train_len=30, target_horizon=1):
        """
        Executes an institutional fail-closed multi-horizon rolling-origin backtest.
        Evaluates U(h) across relevant horizons: h = 1, intermediate, and target_horizon.
        
        Returns strict 4-state release gate:
          - PASS: Evaluated and U(target_horizon) < 1.0.
          - FAIL: Evaluated and U(target_horizon) >= 1.0.
          - NOT_EVALUATED: Insufficient sample length to evaluate target horizon.
          - ERROR: Technical failure during backtest execution.
          
        NEVER FAILS OPEN.
        """
        series = [float(x) for x in series]
        n = len(series)
        target_h = max(1, int(target_horizon))
        
        if n < min_train_len + target_h:
            return {
                "gate_status": "NOT_EVALUATED",
                "passed": False,
                "reason": f"Sample length ({n}) insufficient for rolling-origin evaluation at target horizon h={target_h} (requires at least {min_train_len + target_h} observations).",
                "target_horizon": target_h,
                "theils_u_target": None,
                "hurdles": {}
            }

        horizons_to_test = sorted(list(set([1, min(7, target_h), target_h])))
        hurdles = {}

        try:
            for h in horizons_to_test:
                res_h = self.evaluate_rolling_origin_theils_u(
                    series, forecast_fn, min_train_len=min_train_len, horizon=h
                )
                hurdles[f"h_{h}"] = {
                    "horizon": h,
                    "theils_u": res_h["theils_u"],
                    "hurdle_passed": res_h["hurdle_passed"],
                    "n_evaluations": res_h["n_evaluations"]
                }
            
            target_res = hurdles[f"h_{target_h}"]
            u_target = target_res["theils_u"]
            passed = target_res["hurdle_passed"]
            gate_status = "PASS" if passed else "FAIL"

            return {
                "gate_status": gate_status,
                "passed": passed,
                "target_horizon": target_h,
                "theils_u_target": u_target,
                "hurdles": hurdles,
                "reason": f"U(h={target_h})={u_target:.2f} (< 1.0 hurdle {'PASSED' if passed else 'FAILED'})."
            }
        except Exception as e:
            return {
                "gate_status": "ERROR",
                "passed": False,
                "reason": f"Evaluation crashed: {str(e)}",
                "target_horizon": target_h,
                "theils_u_target": None,
                "hurdles": hurdles
            }

    # Method alias for institutional testing API
    test_granger_causality = granger_causality_test


