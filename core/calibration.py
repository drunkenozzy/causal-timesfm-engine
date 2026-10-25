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
    if not 0 < tau < 1 or not all(math.isfinite(v) for v in [actual, quantile_pred]):
        raise ValueError("Invalid quantile score input")
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
    if not 0 < alpha < 1 or not all(math.isfinite(v) for v in [actual, lower, upper]) or lower > upper:
        raise ValueError("Invalid prediction interval")
    width = upper - lower
    penalty = 0.0
    if actual < lower:
        penalty = (2.0 / alpha) * (lower - actual)
    elif actual > upper:
        penalty = (2.0 / alpha) * (actual - upper)
    return width + penalty

def run_baseline_tournament(series: List[float], h: int = 1, min_train_len: int = 30, **kwargs) -> Dict[str, Any]:
    """Genuine rolling-origin forecasts; no future outcomes enter model selection.

    M4 is an explicitly exploratory log-residual correction, not the legacy
    structural anchor. Its weight is selected from earlier prequential losses
    and stays zero unless it beats BOTH persistence and M1 on those losses.
    Any degraded/invalid origin invalidates the neural tournament, not just
    that row. An injected engine makes offline tests possible.
    """
    from core.engine_timesfm import TimesFmBaselineEngine
    series = [float(v) for v in series]
    if not isinstance(h, int) or h < 1 or min_train_len < 5:
        raise ValueError("Invalid horizon or minimum context")
    if any(not math.isfinite(v) or v <= 0 for v in series):
        raise ValueError("Prices must be finite and positive")
    if len(series) < min_train_len + h:
        return {"status": "INSUFFICIENT_DATA", "n_origins": 0}
    engine = kwargs.get('forecast_engine') or TimesFmBaselineEngine()
    records = []
    weights = (0., .25, .5, 1.)
    for origin in range(min_train_len-1, len(series)-h):
        history = series[:origin+1]
        forecast = engine.forecast(history, horizon_days=h)
        if not forecast.get('timesfm_executed') or forecast.get('degraded_mode', True):
            return dict(status='NOT_EVALUATED_RUNTIME_FAILURE', failed_origin=origin,
                        n_origins=len(records), TIMESFM_COMPARISON_VALID=False)
        p1 = float(forecast['p50_expected'])
        if not math.isfinite(p1) or p1 <= 0:
            return dict(status='NOT_EVALUATED_RUNTIME_FAILURE', failed_origin=origin,
                        n_origins=len(records), TIMESFM_COMPARISON_VALID=False)
        # At origin i, only targets through i have closed.
        known = [r for r in records if r['target_index'] <= origin]
        train = known[-60:]
        bias = sum(math.log(r['actual']/r['m1']) for r in train)/len(train) if len(train)>=20 else 0.
        candidates = {a:p1*math.exp(a*bias) for a in weights}
        validation = [r for r in known if r['training_count']>=20][-30:]
        alpha = 0.
        if len(validation)>=10:
            def loss(key):
                return sum(((r['actual']-r[key])/r['origin_close'])**2 for r in validation)/len(validation)
            losses = {a:sum(((r['actual']-r['candidates'][a])/r['origin_close'])**2 for r in validation)/len(validation) for a in weights}
            best = min(weights, key=lambda a:(losses[a],a))
            if losses[best] < min(loss('m0'),loss('m1')):
                alpha=best
        records.append(dict(origin_index=origin,target_index=origin+h,
            origin_close=history[-1],actual=series[origin+h],m0=history[-1],
            m0b=history[-1]*(history[-1]/history[0])**(h/(len(history)-1)),
            m1=p1,m4=candidates[alpha],alpha=alpha,candidates=candidates,
            training_count=len(train),latest_training_target=max((r['target_index'] for r in train),default=None),
            engine=forecast.get('engine'),timesfm_executed=True,degraded_mode=False))
    models={}
    for label,key in [('M0_Persistence','m0'),('M0b_Drift','m0b'),('TimesFM-Neural','m1'),('M4_RETURN_BIAS_EXPLORATORY','m4')]:
        errs=[r['actual']-r[key] for r in records]
        mspe=sum(e*e for e in errs)/len(errs)
        models[label]=dict(mspe=mspe,rmse=math.sqrt(mspe),mae=sum(abs(e) for e in errs)/len(errs))
    denom=models['M0_Persistence']['rmse']
    for model in models.values():
        model['theils_u2']=model['rmse']/denom if denom else None
    r1=models['TimesFM-Neural']['mspe']; r4=models['M4_RETURN_BIAS_EXPLORATORY']['mspe']
    # Selection and diagnostics are exploratory; no post-hoc confirmatory claim.
    return dict(status='EVALUATED',n_origins=len(records),horizon=h,TIMESFM_COMPARISON_VALID=True,
        evaluation_type='EXPLORATORY_ROLLING_ORIGIN',models=models,origins=records,
        incremental_skill=dict(delta_mspe_m4_vs_m1_pct=100*(1-r4/r1) if r1 else None,
                               structural_inference_status='EXPLORATORY_NOT_CONFIRMATORY'))
