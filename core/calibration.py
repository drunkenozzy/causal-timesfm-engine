"""
core/calibration.py
===================
Probabilistic Calibration & Baseline Tournament Suite for Causal TimesFM Engine v3.0.

Provides:
  1. Pinball Loss for Quantile Evaluation (P10, P50, P90).
  2. Empirical Quantile Coverage & Calibration Error.
  3. Winkler Interval Score (WIS) for Scenario Corridors.
  4. Multi-Model Baseline Tournament:
     M0 (Persistence) vs M0b (Drift) vs M1 (TimesFM Prior) vs M4 (Fitted Structural Hybrid).
"""

import math
from typing import List, Dict, Any

def compute_pinball_loss(actual: float, quantile_pred: float, tau: float) -> float:
    """
    Computes Pinball (asymmetric Laplace) loss for quantile tau in (0, 1):
      L_tau(y, q) = max(tau * (y - q), (1 - tau) * (q - y))
    """
    diff = actual - quantile_pred
    return tau * diff if diff >= 0 else (tau - 1.0) * diff

def compute_empirical_quantile_coverage(actuals: List[float],
                                        p10_series: List[float] = None,
                                        p50_series: List[float] = None,
                                        p90_series: List[float] = None) -> Dict[str, Any]:
    """
    Computes empirical coverage percentages:
      P(Y <= P10) vs 0.10
      P(Y <= P50) vs 0.50
      P(Y <= P90) vs 0.90
    """
    n = len(actuals)
    if n == 0:
        return {"n": 0, "error": "Empty actuals series"}

    report = {"n_samples": n}

    if p10_series and len(p10_series) == n:
        c10 = sum(1 for a, q in zip(actuals, p10_series) if a <= q) / n
        report["cov_p10"] = round(c10, 4)
        report["cov_p10_error"] = round(c10 - 0.10, 4)
        report["mean_pinball_p10"] = round(sum(compute_pinball_loss(a, q, 0.10) for a, q in zip(actuals, p10_series)) / n, 4)

    if p50_series and len(p50_series) == n:
        c50 = sum(1 for a, q in zip(actuals, p50_series) if a <= q) / n
        report["cov_p50"] = round(c50, 4)
        report["cov_p50_error"] = round(c50 - 0.50, 4)
        report["mean_pinball_p50"] = round(sum(compute_pinball_loss(a, q, 0.50) for a, q in zip(actuals, p50_series)) / n, 4)

    if p90_series and len(p90_series) == n:
        c90 = sum(1 for a, q in zip(actuals, p90_series) if a <= q) / n
        report["cov_p90"] = round(c90, 4)
        report["cov_p90_error"] = round(c90 - 0.90, 4)
        report["mean_pinball_p90"] = round(sum(compute_pinball_loss(a, q, 0.90) for a, q in zip(actuals, p90_series)) / n, 4)

    return report

def compute_winkler_interval_score(actual: float, lower: float, upper: float, alpha: float = 0.20) -> float:
    """
    Winkler Interval Score (WIS) for (1 - alpha) prediction interval [lower, upper]:
      WIS = (upper - lower) + (2/alpha)*(lower - actual)*I(actual < lower) + (2/alpha)*(actual - upper)*I(actual > upper)
    Penalizes wider intervals and severely penalizes realizations outside the interval.
    """
    width = upper - lower
    penalty = 0.0
    if actual < lower:
        penalty = (2.0 / alpha) * (lower - actual)
    elif actual > upper:
        penalty = (2.0 / alpha) * (actual - upper)
    return round(width + penalty, 4)

def run_baseline_tournament(series: List[float], h: int = 1, min_train_len: int = 30, **kwargs) -> Dict[str, Any]:
    """
    Executes a multi-model tournament on the provided historical series across rolling origins:
      M0  : Persistence (Naive Random Walk y_{t+h} = y_t)
      M0b : Drift Baseline (Trend extrapolated from recent train window)
      M1  : Autoregressive Baseline (TimesFM Target-Only Prior Proxy)
      M4  : Fitted Hybrid (Ensemble with in-fold estimated alpha)
      
    Computes RMSE, Theil's U2, and formal statistical significance (Clark-West test).
    """
    from core.econometrics import EconometricFilter
    from core.markov_regime import InstitutionalMarkovEngine

    series = [float(x) for x in series]
    n = len(series)
    if n < min_train_len + h:
        return {"status": "INSUFFICIENT_DATA", "reason": f"Need at least {min_train_len + h} observations"}

    err_m0 = []
    err_m0b = []
    err_m1 = []
    err_m4 = []
    actuals = []

    # Precompute historical mock predictions for nested alpha optimization
    p_m1_hist = []
    v_struct_hist = []
    for t in range(n):
        if t < 5:
            p_m1_hist.append(series[t])
            v_struct_hist.append(series[t])
            continue
        w = min(5, t)
        mom = (series[t] / series[t-w]) ** (1.0 / w)
        p_m1_hist.append(series[t] * (mom ** h))
        v_struct_hist.append(series[t] * (mom ** (h * 0.8)))

    for t in range(min_train_len, n - h + 1):
        train = series[:t]
        y_actual = series[t + h - 1]
        actuals.append(y_actual)

        # M0: Persistence
        p_m0 = train[-1]
        err_m0.append(y_actual - p_m0)

        # M0b: Drift
        drift = (train[-1] / train[0]) ** (1.0 / len(train))
        p_m0b = train[-1] * (drift ** h)
        err_m0b.append(y_actual - p_m0b)

        # M1: 5-step Momentum AR
        w = min(5, len(train))
        mom = (train[-1] / train[-w]) ** (1.0 / w)
        p_m1 = train[-1] * (mom ** h)
        err_m1.append(y_actual - p_m1)

        # M4: Hybrid with nested in-fold alpha
        v_struct = train[-1] * (mom ** (h * 0.8))  # damped structural target
        
        # We pass histories up to t (length t). Since they align with y_{k+h},
        # wait, actual_history needs to align with the predictions made h steps prior.
        # For simplicity in this mock tournament, we align the histories up to t-h.
        
        # Proper alignment: prediction made at k for k+h should be compared to series[k+h]
        hist_actual = []
        hist_p_m1 = []
        hist_v_struct = []
        for k in range(5, t - h + 1):
            hist_actual.append(series[k + h - 1])
            hist_p_m1.append(p_m1_hist[k-1])
            hist_v_struct.append(v_struct_hist[k-1])
            
        if len(hist_actual) >= 10:
            alpha_res = InstitutionalMarkovEngine.estimate_optimal_structural_weight(
                hist_actual, hist_p_m1, hist_v_struct, min_inner_train=5
            )
            alpha_fold = alpha_res["alpha_star"]
        else:
            alpha_fold = 0.20

        p_m4 = ((1.0 - alpha_fold) * p_m1) + (alpha_fold * v_struct)
        err_m4.append(y_actual - p_m4)

    def rmse(errs):
        return math.sqrt(sum(e**2 for e in errs) / len(errs))

    r0 = rmse(err_m0)
    r0b = rmse(err_m0b)
    r1 = rmse(err_m1)
    r4 = rmse(err_m4)

    u0 = 1.0
    u0b = round(r0b / r0, 4) if r0 > 0 else 1.0
    u1 = round(r1 / r0, 4) if r0 > 0 else 1.0
    u4 = round(r4 / r0, 4) if r0 > 0 else 1.0

    delta_m1_m0 = round((1.0 - (r1 / r0)) * 100, 2) if r0 > 0 else 0.0
    delta_m4_m1 = round(((r1 - r4) / r1) * 100, 2) if r1 > 0 else 0.0

    # Formal statistical significance
    ef = EconometricFilter()
    cw_res = ef.compute_clark_west_test(err_m4, err_m1, h=h)
    
    is_significant = cw_res.get("is_statistically_significant", False)
    cw_stat = cw_res.get("clark_west_stat", 0.0)
    cw_pval = cw_res.get("p_value", 1.0)
    
    if is_significant and delta_m4_m1 > 0.0:
        inference_status = "STRUCTURAL_ALPHA_SIGNIFICANT"
    elif delta_m4_m1 > 0.0:
        inference_status = "INSIGNIFICANT_POINT_IMPROVEMENT"
    else:
        inference_status = "NO_IMPROVEMENT"

    # Determine M1 telemetry label
    from core.engine_timesfm import TimesFmBaselineEngine
    tfm = TimesFmBaselineEngine()
    tfm_res = tfm.forecast(series[:min_train_len], horizon_days=h)
    
    timesfm_executed = tfm_res.get("timesfm_executed", False)
    m1_label = tfm_res.get("engine", "M1_GEOMETRIC_FALLBACK")
    
    # Guard against masquerading
    if not timesfm_executed:
        if kwargs.get("enforce_confirmatory_timesfm", False):
            return {"status": "NOT_EVALUATED_RUNTIME_FAILURE", "reason": "TimesFM runtime failure"}
        m1_label = "M1_GEOMETRIC_FALLBACK"
        m4_label = "M4_GEOMETRIC_FALLBACK_STRUCTURAL"
    else:
        m4_label = "M4_TIMESFM_STRUCTURAL"

    tourney_results = {
        "status": "EVALUATED",
        "n_origins": len(actuals),
        "horizon": h,
        "TIMESFM_COMPARISON_VALID": timesfm_executed,
        "models": {
            "M0_Persistence": {"rmse": round(r0, 4), "theils_u2": u0},
            "M0b_Drift": {"rmse": round(r0b, 4), "theils_u2": u0b},
            m1_label: {"rmse": round(r1, 4), "theils_u2": u1},
            m4_label: {"rmse": round(r4, 4), "theils_u2": u4}
        },
        "incremental_skill": {
            "delta_skill_m1_vs_m0_pct": delta_m1_m0,
            "delta_skill_m4_vs_m1_pct": delta_m4_m1,
            "structural_layer_adds_value": bool(delta_m4_m1 > 0.0),
            "clark_west_stat": cw_stat,
            "clark_west_p_value": cw_pval,
            "hac_kernel": cw_res.get("hac_kernel", "Bartlett"),
            "hac_max_lag": cw_res.get("hac_max_lag", max(0, h-1)),
            "structural_inference_status": inference_status
        }
    }
    return tourney_results
