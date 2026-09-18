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
        Computes Theil's U2 Statistic (inequality coefficient relative to naive persistence):
          U2 = RMSE(forecast) / RMSE(naive random walk persistence where y_hat_t = y_{t-1})
        Interpretation:
          U2 < 1.0: Model outperforms naive random walk (Hurdle PASSED).
          U2 = 1.0: Model equals naive persistence.
          U2 > 1.0: Model performs worse than naive guess (Hurdle FAILED).
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

    def compute_diebold_mariano_test(self, errors_model, errors_benchmark, h=1, loss_power=2):
        """
        Computes Diebold-Mariano (1995) test with Harvey-Leybourne-Newbold (1997)
        finite-sample correction and Bartlett kernel for overlapping multi-step horizons h.
        Loss differential: d_t = |e_{model, t}|^p - |e_{benchmark, t}|^p (p=2 MSE, p=1 MAE)
        Negative DM indicates model has lower loss than benchmark.
        """
        if len(errors_model) != len(errors_benchmark) or len(errors_model) < 3:
            return {"dm_statistic": 0.0, "p_value": 1.0, "is_statistically_significant": False, "reason": "Insufficient samples"}

        T = len(errors_model)
        h = max(1, int(h))
        if loss_power == 1:
            d = [abs(em) - abs(eb) for em, eb in zip(errors_model, errors_benchmark)]
        else:
            d = [(em ** 2) - (eb ** 2) for em, eb in zip(errors_model, errors_benchmark)]
        d_mean = sum(d) / T

        # Autocovariances up to lag h - 1 with Bartlett weights
        gamma_0 = sum((x - d_mean) ** 2 for x in d) / T
        lr_var = gamma_0

        for k in range(1, min(h, T)):
            gamma_k = sum((d[t] - d_mean) * (d[t - k] - d_mean) for t in range(k, T)) / T
            weight = 1.0 - (k / h)
            lr_var += 2.0 * weight * gamma_k

        se = math.sqrt(max(1e-12, lr_var) / T)
        dm_stat = d_mean / se

        # Harvey-Leybourne-Newbold (1997) finite-sample correction
        hln_factor = math.sqrt(max(1e-6, (T + 1.0 - 2.0 * h + (h * (h - 1.0) / T)) / T))
        dm_hln = dm_stat * hln_factor

        # P-value calculation
        try:
            from scipy.stats import t as t_dist
            p_val = float(2.0 * t_dist.sf(abs(dm_hln), df=max(1, T - 1)))
        except ImportError:
            p_val = float(math.erfc(abs(dm_hln) / math.sqrt(2.0)))

        return {
            "dm_statistic": round(float(dm_hln), 4),
            "p_value": round(float(p_val), 4),
            "is_statistically_significant": bool(p_val < 0.05),
            "model_is_superior": bool(dm_hln < 0 and p_val < 0.05),
            "horizon": h,
            "loss_differential_mean": round(float(d_mean), 6)
        }

    def compute_clark_west_test(self, errors_model, errors_benchmark, h=1):
        """
        Computes the Clark-West (2007) test for nested models.
        M_benchmark (null) is nested within M_model (alternative).
        Adjusts the MSFE difference to account for parameter estimation noise under the null.
        """
        T = len(errors_model)
        if T < 2 or len(errors_benchmark) != T:
            return {"clark_west_stat": 0.0, "p_value": 1.0, "is_statistically_significant": False, "adjusted_msfe_diff": 0.0}

        # CW adjustment: f_t = err_bench^2 - (err_model^2 - (err_bench - err_model)^2)
        f_t = []
        for em, eb in zip(errors_model, errors_benchmark):
            f_t.append((eb ** 2) - ((em ** 2) - ((eb - em) ** 2)))

        mean_f = sum(f_t) / T
        
        # Sample long-run variance of f_t (HAC Bartlett Kernel for overlapping forecasts)
        lr_var = sum((x - mean_f) ** 2 for x in f_t) / T
        
        if h > 1:
            for k in range(1, int(h)):
                cov_k = sum((f_t[t] - mean_f) * (f_t[t - k] - mean_f) for t in range(k, T)) / T
                weight = 1.0 - (k / h)
                lr_var += 2.0 * weight * cov_k
                
        if lr_var <= 1e-12:
            return {"clark_west_stat": 0.0, "p_value": 1.0, "is_statistically_significant": False, "adjusted_msfe_diff": mean_f}

        cw_stat = mean_f / math.sqrt(lr_var / T)
        
        # One-sided test (alternative is M_model is better)
        try:
            from scipy.stats import norm
            p_val = float(norm.sf(cw_stat))
        except ImportError:
            p_val = float(0.5 * math.erfc(cw_stat / math.sqrt(2.0)))
        
        return {
            "clark_west_stat": round(float(cw_stat), 4),
            "p_value": round(float(p_val), 4),
            "is_statistically_significant": bool(p_val < 0.05),
            "adjusted_msfe_diff": round(float(mean_f), 6)
        }

    def compute_block_bootstrap_theils_u(self, errors_model, errors_benchmark, h=1, n_boot=200, block_size=None):
        """
        Computes Moving Block Bootstrap (MBB) 95% confidence intervals for Theil's U2
        and paired loss differentials.
        Preserves serial dependence in overlapping multi-step forecast errors.
        Imposes the null hypothesis by recentering the paired loss differentials.
        """
        import random
        T = len(errors_model)
        if T < 4:
            return {
                "theils_u2_point": 1.0, "ci_95_lower": 1.0, "ci_95_upper": 1.0, 
                "diff_ci_95_lower": 0.0, "diff_ci_95_upper": 0.0, 
                "superiority_established": False, "bootstrap_p_value": 1.0,
                "n_bootstrap": n_boot, "block_length": 1, "CI_method": "percentile"
            }

        b = block_size if block_size else max(2, min(int(h), T // 3))
        rmse_m = math.sqrt(sum(e**2 for e in errors_model) / T)
        rmse_b = math.sqrt(sum(e**2 for e in errors_benchmark) / T)
        u_point = rmse_m / rmse_b if rmse_b > 1e-12 else 1.0

        diffs = [(errors_benchmark[t]**2) - (errors_model[t]**2) for t in range(T)]
        mean_diff = sum(diffs) / T
        
        # Recenter differentials under the null hypothesis (mean = 0)
        recentered_diffs = [d - mean_diff for d in diffs]

        boot_u = []
        boot_diff_c = []  # Recentered bootstrap means (for p-value)
        boot_diff = []    # Raw bootstrap means (for CI)
        
        n_blocks = max(1, (T + b - 1) // b)

        rng = random.Random(42)  # Deterministic seed for reproducible testing
        for _ in range(n_boot):
            sample_m_sq = []
            sample_b_sq = []
            sample_diff_c = []
            for _ in range(n_blocks):
                start_idx = rng.randint(0, max(0, T - b))
                for idx in range(start_idx, min(T, start_idx + b)):
                    sample_m_sq.append(errors_model[idx] ** 2)
                    sample_b_sq.append(errors_benchmark[idx] ** 2)
                    sample_diff_c.append(recentered_diffs[idx])
            
            sample_m_sq = sample_m_sq[:T]
            sample_b_sq = sample_b_sq[:T]
            sample_diff_c = sample_diff_c[:T]
            
            mean_m = sum(sample_m_sq) / len(sample_m_sq)
            mean_b = sum(sample_b_sq) / len(sample_b_sq)
            
            r_m = math.sqrt(mean_m)
            r_b = math.sqrt(mean_b)
            boot_u.append(r_m / r_b if r_b > 1e-12 else 1.0)
            
            # Differential CI (Model vs Bench, using raw differentials)
            # Actually, mean_m - mean_b is -(mean_bench - mean_model)
            # Wait, earlier code did mean_m - mean_b. If M is better, mean_m < mean_b, diff < 0.
            # Let's keep the sign convention: diff = mean_m - mean_b (Negative means M is better).
            boot_diff.append(mean_m - mean_b)
            
            # Bootstrap p-value for H0: M is not better than Bench (mean_diff <= 0)
            # which is equivalent to mean_m - mean_b >= 0.
            # So under H0, we check how often the recentered (mean_m_c - mean_b_c) is <= observed (mean_m - mean_b).
            # We recentered diff = bench - model. So M is better if diff > 0.
            # Let's use the recentered bench - model:
            boot_diff_c.append(sum(sample_diff_c) / len(sample_diff_c))

        boot_u.sort()
        boot_diff.sort()
        idx_low = int(0.025 * len(boot_u))
        idx_high = int(0.975 * len(boot_u))
        
        ci_low = boot_u[idx_low]
        ci_high = boot_u[min(len(boot_u) - 1, idx_high)]
        
        diff_ci_low = boot_diff[idx_low]
        diff_ci_high = boot_diff[min(len(boot_diff) - 1, idx_high)]

        # Superiority is established if the upper bound of the differential is < 0
        superiority = bool(diff_ci_high < 0.0)
        
        # P-value: H0 is that true mean_diff <= 0.
        # We observed mean_diff. What fraction of recentered boot_diff_c > mean_diff?
        p_val = sum(1 for dc in boot_diff_c if dc >= mean_diff) / len(boot_diff_c) if mean_diff > 0 else 1.0

        return {
            "theils_u2_point": round(float(u_point), 4),
            "ci_95_lower": round(float(ci_low), 4),
            "ci_95_upper": round(float(ci_high), 4),
            "diff_ci_95_lower": round(float(diff_ci_low), 6),
            "diff_ci_95_upper": round(float(diff_ci_high), 6),
            "superiority_established": superiority,
            "bootstrap_p_value": round(float(p_val), 4),
            "n_bootstrap": n_boot,
            "block_length": b,
            "random_seed": 42,
            "CI_method": "percentile"
        }

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
            "theils_u2": float,
            "hurdle_passed": bool,
            "n_evaluations": int,
            "forecast_rmse": float,
            "naive_rmse": float,
            "horizon": int,
            "errors_model": list,
            "errors_naive": list
          }
        """
        series = [float(x) for x in series]
        n = len(series)
        if n < min_train_len + horizon:
            raise ValueError(f"Series length ({n}) insufficient for rolling-origin evaluation with min_train_len={min_train_len} and horizon={horizon}.")

        errors_model = []
        errors_naive = []

        for t in range(min_train_len, n - horizon + 1):
            train_history = series[:t]
            actual_val = series[t + horizon - 1]
            naive_pred = series[t - 1]

            pred_val = forecast_fn(train_history, horizon)
            if isinstance(pred_val, dict):
                pred_val = pred_val.get("p50_expected", pred_val.get("expected_target", pred_val.get("reconciled_p50", train_history[-1])))

            errors_model.append(float(pred_val) - actual_val)
            errors_naive.append(naive_pred - actual_val)

        m_fc = sum(e ** 2 for e in errors_model) / len(errors_model)
        m_nv = sum(e ** 2 for e in errors_naive) / len(errors_naive)

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
            "theils_u2": theils_u,
            "hurdle_passed": hurdle_passed,
            "n_evaluations": len(errors_model),
            "forecast_rmse": round(rmse_fc, 4),
            "naive_rmse": round(rmse_nv, 4),
            "horizon": horizon,
            "errors_model": errors_model,
            "errors_naive": errors_naive
        }

    def evaluate_multi_horizon_theils_u(self, series, forecast_fn, min_train_len=30, target_horizon=1, min_oos_origins=1):
        """
        Executes an institutional fail-closed multi-horizon rolling-origin backtest.
        Evaluates U2(h) across relevant horizons: h = 1, intermediate, and target_horizon.
        Enforces sample sufficiency based on forecast origins: requires n_origins >= min_oos_origins.
        Computes Diebold-Mariano test and Moving Block Bootstrap 95% confidence intervals.
        
        Returns strict 4-state release gate:
          - PASS: Evaluated and U2(target_horizon) < 1.0.
          - FAIL: Evaluated and U2(target_horizon) >= 1.0.
          - NOT_EVALUATED: Insufficient sample length / origins to evaluate target horizon.
          - ERROR: Technical failure during backtest execution.
          
        NEVER FAILS OPEN.
        """
        series = [float(x) for x in series]
        n = len(series)
        target_h = max(1, int(target_horizon))
        n_origins = n - min_train_len - target_h + 1
        
        if n < min_train_len + target_h or n_origins < min_oos_origins:
            return {
                "gate_status": "NOT_EVALUATED",
                "passed": False,
                "reason": f"Insufficient forecast origins ({max(0, n_origins)} < {min_oos_origins} required) for target horizon h={target_h}.",
                "target_horizon": target_h,
                "theils_u_target": None,
                "n_origins": max(0, n_origins),
                "hurdles": {}
            }

        horizons_to_test = sorted(list(set([1, min(7, target_h), target_h])))
        hurdles = {}

        try:
            for h in horizons_to_test:
                res_h = self.evaluate_rolling_origin_theils_u(
                    series, forecast_fn, min_train_len=min_train_len, horizon=h
                )
                
                # Run Diebold-Mariano test and block bootstrap on target horizon
                dm_res = self.compute_diebold_mariano_test(res_h["errors_model"], res_h["errors_naive"], h=h)
                boot_res = self.compute_block_bootstrap_theils_u(res_h["errors_model"], res_h["errors_naive"], h=h)

                hurdles[f"h_{h}"] = {
                    "horizon": h,
                    "theils_u": res_h["theils_u"],
                    "theils_u2": res_h["theils_u"],
                    "hurdle_passed": res_h["hurdle_passed"],
                    "n_evaluations": res_h["n_evaluations"],
                    "dm_test": dm_res,
                    "bootstrap_ci": boot_res
                }
            
            target_res = hurdles[f"h_{target_h}"]
            u_target = target_res["theils_u"]
            passed = target_res["hurdle_passed"]
            gate_status = "PASS" if passed else "FAIL"

            dm_target = target_res["dm_test"]
            boot_target = target_res["bootstrap_ci"]
            stat_sig = dm_target.get("is_statistically_significant", False)
            boot_superior = boot_target.get("superiority_established", False)

            return {
                "gate_status": gate_status,
                "passed": passed,
                "target_horizon": target_h,
                "theils_u_target": u_target,
                "diebold_mariano": dm_target,
                "bootstrap_ci": boot_target,
                "statistical_superiority_established": bool(passed and (stat_sig or boot_superior)),
                "hurdles": hurdles,
                "reason": f"U2(h={target_h})={u_target:.2f} (< 1.0 hurdle {'PASSED' if passed else 'FAILED'}). 95% CI: [{boot_target['ci_95_lower']:.2f}, {boot_target['ci_95_upper']:.2f}]."
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


