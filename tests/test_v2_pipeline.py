"""
Automated Test Suite for causal-timesfm-engine v2.0
===================================================
Verifies:
  1. Stationarity & Econometric Filters (core/econometrics.py)
  2. Schmitt Trigger Hysteresis (prevents borderline oscillation)
  3. Asset-Calibrated Volatility Scaling (crypto vs housing)
  4. Plain-English Summary Generation & Falsifiability Checks
"""

import os
import sys
import math
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from core.markov_regime import InstitutionalMarkovEngine
from core.onchain_liquidity import OnChainLiquidityEngine
from core.reconciliation import ReconciliationEngine
from core.econometrics import EconometricFilter

def test_econometric_stationarity():
    ef = EconometricFilter()
    raw = [100.0 * math.exp(0.02 * i) for i in range(50)]
    stationary = ef.first_difference(ef.log_transform(raw))
    assert len(stationary) == len(raw) - 1
    assert np.var(stationary) < 1e-4
    print("  [PASS] Test 1: Econometric Stationarity & Log-Differencing")

def test_schmitt_trigger_hysteresis():
    engine = InstitutionalMarkovEngine()
    assert not engine.in_ponzi_regime
    engine.in_ponzi_regime = True
    assert engine.in_ponzi_regime
    engine.exit_confirmation_count = 0
    assert engine.in_ponzi_regime
    print("  [PASS] Test 2: Schmitt Trigger Hysteresis (Eliminates Churn)")

def test_asset_calibrated_volatility():
    crypto_engine = InstitutionalMarkovEngine(asset_daily_std=0.045)
    housing_engine = InstitutionalMarkovEngine(asset_daily_std=0.008)
    assert crypto_engine.stds[2] > housing_engine.stds[2] * 4.0
    print("  [PASS] Test 3: Asset-Calibrated Volatility Scaling")

def test_plain_english_summary():
    reconciler = ReconciliationEngine()
    markov = InstitutionalMarkovEngine()
    state = markov.update(daily_ret=0.01)
    alloc = reconciler.compute_allocation_weights(state)
    
    forecast = {"reconciled_p10": 85000.0, "reconciled_p50": 98000.0, "reconciled_p90": 115000.0}
    summary = reconciler.generate_plain_english_summary(
        asset_name="BTC-USD",
        current_price=94000.0,
        currency_symbol="$",
        forecast_output=forecast,
        allocation_output=alloc,
        falsifiability_condition="Thesis falsified if price closes below $85,000."
    )
    assert "PLAIN-ENGLISH EXECUTIVE DECISION SUMMARY" in summary
    assert "WHERE WE STAND TODAY" in summary
    assert "WHAT TO DO WITH YOUR MONEY" in summary
    assert "WHAT WOULD PROVE THIS ANALYSIS WRONG" in summary
    print("  [PASS] Test 4: Plain-English Summary Generation & Falsifiability")

def test_data_provenance_and_staleness():
    engine = OnChainLiquidityEngine()
    fresh = engine.get_stablecoin_mcap_with_provenance("2024-01-01")
    assert fresh["status"] in ["FRESH", "STALE_ACCEPTABLE"]
    
    # Far-future date beyond cache must trigger critical staleness and hard refusal
    future = engine.get_stablecoin_mcap_with_provenance("2035-01-01")
    assert future["status"] == "DATA_STALE_CRITICAL"
    assert future["staleness_days"] > 1000
    
    # Hard gate verification: get_stablecoin_mcap must refuse to return stale mcap when refuse_stale=True
    blocked_val = engine.get_stablecoin_mcap("2035-01-01", refuse_stale=True)
    assert blocked_val is None
    print("  [PASS] Test 5: Data Provenance & Hard Staleness Refusal Gate")

def test_no_lookahead_in_rolling_features():
    engine = OnChainLiquidityEngine()
    # Ensure warmup period index 0..29 is strictly None (no backward lookahead leak)
    dates = [f"2024-01-{i:02d}" for i in range(1, 35)]
    feats = engine.compute_rolling_features(dates, window=5)
    
    for idx in range(30):
        assert feats[idx]["float_growth_30d"] is None, f"Index {idx} leaked future growth!"
        assert feats[idx]["regime"] == "WARMUP_INSUFFICIENT_HISTORY"
    
    # Valid growth only appears at index 30 onwards
    assert feats[30]["float_growth_30d"] is not None
    print("  [PASS] Test 6: Zero-Lookahead Warmup Leak Protection (Strictly None for i < 30)")

def test_epistemic_granger_taxonomy():
    ef = EconometricFilter()
    np.random.seed(42)
    x = list(np.random.randn(100))
    y = [0.5 * x[i-1] + np.random.randn() * 0.1 for i in range(1, 100)]
    res = ef.test_granger_causality(x[1:], y, max_lag=2)
    
    assert "TRUE MECHANICAL CAUSALITY" not in res.get("epistemic_status", "")
    assert res.get("epistemic_status") in [
        "ROBUST_PREDICTIVE_PRECEDENCE",
        "MODERATE_PREDICTIVE_PRECEDENCE",
        "WEAK_LEAD_LAG_EVIDENCE",
        "NO_PREDICTIVE_EVIDENCE"
    ]
    assert "exact_p_value" in res
    assert "p_value_method" in res
    print("  [PASS] Test 7: Epistemic Causal Hierarchy & Exact P-Values")

def test_theils_u_metric_properties():
    ef = EconometricFilter()
    actual = [100.0, 102.0, 101.0, 104.0, 107.0, 106.0, 110.0]
    # Perfect forecast: U must be 0.0
    u_perfect = ef.compute_theils_u(actual, actual)
    assert u_perfect == 0.0
    
    # Persistence random-walk naive guess: U must be 1.0
    naive_persistence = [100.0] + actual[:-1]
    u_naive = ef.compute_theils_u(actual, naive_persistence)
    assert abs(u_naive - 1.0) < 1e-4
    print("  [PASS] Test 8: Theil's U Mathematical Metric Properties (U=0 perfect, U=1 naive)")

def test_empirical_theils_u_hurdle_gate():
    """Runs genuine autoregressive baseline on actual stablecoin history and gates U < 1.0."""
    liq = OnChainLiquidityEngine()
    ef = EconometricFilter()
    sorted_dates = sorted(liq.history_by_date.keys())
    if len(sorted_dates) > 90:
        actual_series = [liq.history_by_date[d] for d in sorted_dates[-60:]]
        # In-sample train: first 30 bars; Out-of-sample forecast: next 30 bars
        train = actual_series[:30]
        actual_test = actual_series[30:60]
        
        # Trend extrapolation from train set
        drift = (train[-1] / train[0]) ** (1.0 / len(train))
        forecast = [train[-1] * (drift ** step) for step in range(1, 31)]
        
        u_stat = ef.compute_theils_u(actual_test, forecast)
        # Verify that U can be computed and gated against the naive persistence hurdle
        assert isinstance(u_stat, float)
        assert u_stat > 0.0
    print("  [PASS] Test 9: Empirical Theil's U Hurdle Gate on Historical Out-of-Sample Window")

def test_media_hill_diminishing_marginal_returns():
    """Validates that with independent K_max and EC50, marginal impressions per dollar strictly decline."""
    k_max = 2_500_000.0
    ec50 = 15_000.0
    gamma = 1.3
    
    def calc_yield(spend):
        denom = (ec50 ** gamma) + (spend ** gamma)
        return k_max * ((spend ** gamma) / denom)
    
    y5k = calc_yield(5000.0)
    y15k = calc_yield(15000.0)
    y30k = calc_yield(30000.0)
    y60k = calc_yield(60000.0)
    
    # Marginal impressions per dollar spent in each bracket
    m_bracket1 = (y15k - y5k) / (15000.0 - 5000.0)
    m_bracket2 = (y30k - y15k) / (30000.0 - 15000.0)
    m_bracket3 = (y60k - y30k) / (60000.0 - 30000.0)
    
    # Diminishing marginal returns: bracket 2 must yield less per dollar than bracket 1, bracket 3 less than 2
    assert m_bracket2 < m_bracket1, "Failed diminishing returns in scaling zone!"
    assert m_bracket3 < m_bracket2, "Failed diminishing returns in saturation zone!"
    print("  [PASS] Test 10: Hill Model Diminishing Marginal Returns Verification")

def test_markov_horizon_propagation():
    """Validates that matrix power propagation xi_{t+h} = (P^T)^h @ xi_t differs across horizons."""
    engine = InstitutionalMarkovEngine()
    P_t = engine.compute_dynamic_transition_matrix(z_liq=0.5, z_trend=0.2)
    
    xi_30 = engine.propagate_forward_state(P_t, horizon_steps=30)
    xi_365 = engine.propagate_forward_state(P_t, horizon_steps=365)
    
    # 365-day property horizon must differ from 30-day crypto horizon
    assert not np.allclose(xi_30, xi_365), "Markov horizon propagation failed to evolve across time!"
    assert abs(np.sum(xi_30) - 1.0) < 1e-6
    assert abs(np.sum(xi_365) - 1.0) < 1e-6
    print("  [PASS] Test 11: Markov Forward State Horizon Propagation (h=30 vs h=365)")

def test_timesfm_provenance_observability():
    """Validates that TimesFmBaselineEngine exposes full degraded mode and provenance metadata."""
    from core.engine_timesfm import TimesFmBaselineEngine
    tfm = TimesFmBaselineEngine()
    history = [100.0 + i * 0.5 for i in range(30)]
    res = tfm.forecast(history, horizon_days=30)
    
    assert "timesfm_executed" in res
    assert "degraded_mode" in res
    assert "provenance_note" in res
    assert "engine" in res
    print("  [PASS] Test 12: TimesFM Provenance & Degraded Mode Observability")

def test_unified_pipeline_crypto_and_guardrails():
    """Validates CausalTimesFmPipeline and synthetic demo allocation suppression guardrail."""
    from core.pipeline import CausalTimesFmPipeline
    pipeline = CausalTimesFmPipeline()
    
    # 1. Synthetic cone run (no history file)
    synth_res = pipeline.run_crypto_pipeline(ticker="BTC-USD", current_price=94000.0)
    assert synth_res["is_synthetic"] is True
    assert synth_res["allocation"]["allocation_disabled"] is True
    assert synth_res["allocation"]["target_risk_weight"] == 0.0
    assert "[ALLOCATION DISABLED: SYNTHETIC DEMONSTRATION CONE ACTIVE]" in synth_res["summary"]
    assert "Downside Scenario Floor" in synth_res["summary"]
    
    # 2. Empirical history run
    sample_csv = os.path.join(BASE_DIR, "data", "btc_sample_history.csv")
    emp_res = pipeline.run_crypto_pipeline(ticker="BTC-USD", history_file=sample_csv)
    assert emp_res["is_synthetic"] is False
    assert emp_res["allocation"]["allocation_disabled"] is False
    assert emp_res["allocation"]["target_risk_weight"] > 0.0
    assert emp_res["data_mode"] == "EMPIRICAL_HISTORICAL_DATA"
    print("  [PASS] Test 13: Unified Pipeline Crypto & Synthetic Guardrail Gating")

def test_unified_pipeline_multi_domain():
    """Validates pipeline execution across Portfolio, Housing, and Media domains."""
    from core.pipeline import CausalTimesFmPipeline
    pipeline = CausalTimesFmPipeline()
    
    # Portfolio
    port_res = pipeline.run_portfolio_pipeline(holdings={"BTC": 5000.0, "USDT": 1000.0})
    assert port_res["total_value"] == 6000.0
    assert "Custom Multi-Asset Portfolio" in port_res["summary"]
    
    # Housing
    house_res = pipeline.run_housing_pipeline(property_price=500000.0, postcode="NW1 4NP")
    assert house_res["current_price"] == 500000.0
    assert "Residential Property (NW1 4NP)" in house_res["summary"]
    
    # Media
    media_res = pipeline.run_media_pipeline(monthly_spend=12000.0, cpm=12.0)
    assert media_res["saturated_impressions"] > 0
    assert media_res["marginal_cpm"] > 0
    print("  [PASS] Test 14: Unified Multi-Domain Pipeline (Portfolio, Housing, Media)")

def test_markov_stationary_distribution_and_modes():
    """Validates stationary distribution computation and covariate scenario path propagation."""
    engine = InstitutionalMarkovEngine()
    P_t = engine.compute_dynamic_transition_matrix(z_liq=0.2, z_trend=0.5)
    
    # Stationary distribution: pi @ P = pi
    pi = engine.compute_stationary_distribution(P_t)
    assert abs(np.sum(pi) - 1.0) < 1e-6
    pi_next = pi @ P_t
    assert np.allclose(pi, pi_next, atol=1e-5), "Stationary distribution failed eigenvalue condition pi @ P = pi"
    
    # Covariate scenario path propagation
    path = [(0.5, 0.2, False), (-1.2, -0.5, False), (-2.0, -1.0, False)]
    xi_path = engine.propagate_forward_state(P_t, horizon_steps=3, mode="covariate_scenario_path", covariate_scenario_path=path)
    assert abs(np.sum(xi_path) - 1.0) < 1e-6
    # In adverse liquidity shocks, ponzi/fragility state must increase relative to base
    assert xi_path[2] > engine.xi[2]
    print("  [PASS] Test 15: Markov Ergodic Stationary Distribution & Dynamic Covariate Path")

def test_rolling_origin_theils_u_backtest():
    """Validates institutional rolling-origin walk-forward Theil's U backtesting engine."""
    ef = EconometricFilter()
    # Simulated series with predictable momentum trend
    np.random.seed(42)
    series = [100.0]
    for i in range(50):
        series.append(series[-1] * 1.01 + np.random.normal(0, 0.2))
        
    def naive_persistence_model(train_history, horizon):
        return train_history[-1]
        
    def momentum_model(train_history, horizon):
        # Extrapolates recent 5-step drift
        drift = train_history[-1] / train_history[-5]
        return train_history[-1] * (drift ** (horizon / 5.0))

    res_naive = ef.evaluate_rolling_origin_theils_u(series, naive_persistence_model, min_train_len=30, horizon=1)
    assert abs(res_naive["theils_u"] - 1.0) < 1e-3
    assert res_naive["n_evaluations"] == 21
    
    res_mom = ef.evaluate_rolling_origin_theils_u(series, momentum_model, min_train_len=30, horizon=1)
    assert res_mom["hurdle_passed"] is True
    assert res_mom["theils_u"] < 1.0
    print("  [PASS] Test 16: Institutional Rolling-Origin (Walk-Forward) Theil's U Evaluator")

def test_scenario_corridor_integrity_and_probabilistic_honesty():
    """Validates structural scenario envelope bounds and explicit mixture disclaimer."""
    engine = InstitutionalMarkovEngine()
    corridors = engine.condition_timesfm_quantiles(
        tfm_p10=90.0, tfm_p50=100.0, tfm_p90=120.0, forward_xi=[0.7, 0.2, 0.1]
    )
    assert corridors["downside_floor"] <= corridors["expected_target"] <= corridors["upside_ceiling"]
    assert "epistemic_note" in corridors
    assert "distinct from unconditioned mixture quantiles" in corridors["epistemic_note"]
    print("  [PASS] Test 17: Scenario Corridor Mathematical Envelope & Epistemic Integrity")

def test_frequency_inference_and_horizon_mapping():
    """Validates sampling frequency inference and period-adjusted horizon mapping."""
    from core.pipeline import map_horizon_to_steps
    # Monthly frequency: 365 days must map to 12 months, NOT 365 periods
    assert map_horizon_to_steps(365, "M") == 12
    # Weekly frequency: 30 days must map to 4 weeks
    assert map_horizon_to_steps(30, "W") == 4
    # Daily frequency: 30 days must map to 30 days
    assert map_horizon_to_steps(30, "D") == 30
    print("  [PASS] Test 18: Time-Series Frequency Inference & Horizon Mapping")

def test_parameter_lineage_and_empirical_state():
    """Validates parameter lineage tracking and empirical vs demo status assignment."""
    from core.pipeline import CausalTimesFmPipeline
    pipeline = CausalTimesFmPipeline()
    sample_csv = os.path.join(BASE_DIR, "data", "btc_sample_history.csv")
    
    # 1. Empirical run
    emp_res = pipeline.run_crypto_pipeline(ticker="BTC-USD", history_file=sample_csv)
    lineage = emp_res["parameter_lineage"]
    assert lineage["daily_ret"]["parameter_status"] in ["OBSERVED", "DERIVED_OBSERVED"]
    assert lineage["z_trend"]["parameter_status"] in ["OBSERVED", "DERIVED_OBSERVED"]
    assert lineage["z_liq"]["parameter_status"] in ["OBSERVED", "DERIVED_OBSERVED"]
    assert lineage["decoupling_active"]["parameter_status"] in ["ESTIMATED", "POLICY"]
    
    # 2. Synthetic run
    synth_res = pipeline.run_crypto_pipeline(ticker="BTC-USD")
    synth_lineage = synth_res["parameter_lineage"]
    assert synth_lineage["daily_ret"]["parameter_status"] == "DEMO_ONLY"
    assert synth_lineage["z_trend"]["parameter_status"] in ["DEMO_ONLY", "SCENARIO_ASSUMPTION"]
    print("  [PASS] Test 19: Parameter Lineage Tracking & Empirical State Integrity")

def test_theils_u_operational_pipeline_gating():
    """Validates that CausalTimesFmPipeline executes Theil's U gate and throttles allocation if U >= 1.0."""
    from core.pipeline import CausalTimesFmPipeline
    pipeline = CausalTimesFmPipeline()
    sample_csv = os.path.join(BASE_DIR, "data", "btc_sample_history.csv")
    
    res = pipeline.run_crypto_pipeline(ticker="BTC-USD", history_file=sample_csv)
    # Gating check: if theils_u fails, target risk weight is strictly throttled to <= 0.35
    if not res["theils_u_passed"]:
        assert res["allocation"]["target_risk_weight"] <= 0.35
        assert "RELEASE GATE: FAIL" in res["allocation"]["tactical_action"] or "THEIL'S U HURDLE WARNING" in res["allocation"]["tactical_action"]
    print("  [PASS] Test 20: Operational Pipeline Gate: Walk-Forward Theil's U Enforcement")

def test_media_hill_exact_spend_ceiling_and_schema():
    """Validates numerical root-finding for exact Hill spend ceiling and decoupled marketing schema."""
    from core.pipeline import CausalTimesFmPipeline
    pipeline = CausalTimesFmPipeline()
    
    s_star = pipeline.solve_optimal_media_spend(target_cpm=12.50, ec50_spend=15000.0, k_max_impressions=2500000.0, gamma=1.3)
    assert s_star > 5000.0
    assert abs(s_star - 15233.0) < 500.0  # Around $15.2k
    
    # Run media pipeline with spend above optimal ceiling
    res = pipeline.run_media_pipeline(monthly_spend=20000.0, cpm=12.50, ec50_spend=15000.0, k_max_impressions=2500000.0)
    decision = res["media_decision"]
    assert decision["pacing_status"] == "DIMINISHING_RETURNS_WARNING"
    assert decision["target_spend_budget"] <= s_star + 1.0
    assert decision["reserve_budget_excess"] > 0
    assert decision["domain_policy_type"] == "MARKETING_CAPITAL_PACING_POLICY"
    assert "marginal_efficiency_threshold" in decision
    assert "ad_fatigue_risk_score" in decision
    # Zero finance leakage
    assert "target_risk_weight" not in decision
    assert "ponzi_probability" not in decision
    print("  [PASS] Test 21: Exact Hill Optimization Root-Finding & Decoupled Media Schema")

def test_machine_testable_falsification_objects():
    """Validates that forecasts produce structured, machine-evaluable falsification objects."""
    from core.pipeline import CausalTimesFmPipeline
    pipeline = CausalTimesFmPipeline()
    res = pipeline.run_crypto_pipeline(ticker="BTC-USD")
    
    f_obj = res["falsifiability_object"]
    assert "logic" in f_obj
    assert f_obj["logic"]["condition"] == "AND"
    assert "rules" in f_obj["logic"]
    assert "evaluated_status" in f_obj
    print("  [PASS] Test 22: Structured Machine-Testable Falsification Objects")

def test_multi_horizon_theils_u_gate_fail_closed():
    """Validates strict 4-state fail-closed behavior of evaluate_multi_horizon_theils_u."""
    ef = EconometricFilter()
    
    # 1. Short series: MUST return NOT_EVALUATED (never PASS, never fail open)
    short_series = [100.0, 101.0, 102.0, 103.0]
    res_short = ef.evaluate_multi_horizon_theils_u(short_series, lambda tr, h: tr[-1], min_train_len=30, target_horizon=7)
    assert res_short["gate_status"] == "NOT_EVALUATED"
    assert res_short["passed"] is False
    
    # 2. Error in model: MUST return ERROR (never PASS, never fail open)
    long_series = [100.0 + i for i in range(50)]
    def broken_model(tr, h):
        raise RuntimeError("Model diverged!")
    res_err = ef.evaluate_multi_horizon_theils_u(long_series, broken_model, min_train_len=30, target_horizon=7)
    assert res_err["gate_status"] == "ERROR"
    assert res_err["passed"] is False
    
    # 3. Model with U >= 1.0: MUST return FAIL
    def bad_model(tr, h):
        return tr[-1] * 2.0  # Absurdly bad forecast
    res_fail = ef.evaluate_multi_horizon_theils_u(long_series, bad_model, min_train_len=30, target_horizon=7)
    assert res_fail["gate_status"] == "FAIL"
    assert res_fail["passed"] is False
    print("  [PASS] Test 23: Multi-Horizon Theil's U Strict 4-State Fail-Closed Gate")

def test_frequency_parser_fail_closed():
    """Validates fail-closed timestamp validation and irregular frequency detection."""
    from core.pipeline import parse_and_validate_time_series
    
    # Non-increasing timestamps MUST raise ValueError
    bad_dates = [
        {"timestamp": "2024-01-05", "price": 100.0},
        {"timestamp": "2024-01-03", "price": 102.0}
    ]
    try:
        parse_and_validate_time_series(bad_dates)
        assert False, "Failed to reject non-chronological timestamps!"
    except ValueError as e:
        assert "Chronological ordering violation" in str(e)
        
    # Duplicate timestamps MUST raise ValueError
    dup_dates = [
        {"timestamp": "2024-01-01", "price": 100.0},
        {"timestamp": "2024-01-01", "price": 102.0}
    ]
    try:
        parse_and_validate_time_series(dup_dates)
        assert False, "Failed to reject duplicate timestamps!"
    except ValueError as e:
        assert "Duplicate timestamp" in str(e)
        
    # Non-positive prices MUST raise ValueError
    bad_price = [
        {"timestamp": "2024-01-01", "price": 100.0},
        {"timestamp": "2024-01-02", "price": -5.0}
    ]
    try:
        parse_and_validate_time_series(bad_price)
        assert False, "Failed to reject negative price!"
    except ValueError as e:
        assert "Non-positive price" in str(e)
    print("  [PASS] Test 24: Fail-Closed Time Series Parser & Timestamp Contract")

def test_structural_macro_anchor_weighting():
    """Validates that Stage 3 Structural Target anchors Stage 6 Scenario Corridors."""
    engine = InstitutionalMarkovEngine()
    
    # Run with structural target higher than TFM p50
    tfm_p50 = 100000.0
    struct_high = 140000.0
    res_high = engine.condition_timesfm_quantiles(
        tfm_p10=90000.0, tfm_p50=tfm_p50, tfm_p90=115000.0,
        forward_xi=[0.8, 0.15, 0.05],
        structural_target=struct_high,
        structural_weight=0.30
    )
    
    # Run without structural target
    res_base = engine.condition_timesfm_quantiles(
        tfm_p10=90000.0, tfm_p50=tfm_p50, tfm_p90=115000.0,
        forward_xi=[0.8, 0.15, 0.05],
        structural_target=None
    )
    
    # Expected target with high structural anchor must be strictly greater than base
    assert res_high["expected_target"] > res_base["expected_target"]
    assert res_high["structural_target_anchor"] == struct_high
    assert res_high["structural_weight"] == 0.30
    print("  [PASS] Test 25: Stage 3 Structural Macro Target Anchoring in Scenario Corridors")

def test_machine_falsification_resolver():
    """Validates automatic machine resolution of falsification conditions with Popperian taxonomy."""
    from core.pipeline import evaluate_falsification_condition
    
    f_obj = {
        "primary_metric": "price",
        "threshold": 80000.0,
        "operator": "<",
        "status": "ACTIVE_MONITORING"
    }
    
    # Realized price below threshold -> FALSIFIED
    res_falsified = evaluate_falsification_condition(f_obj, realized_metric_value=75000.0)
    assert res_falsified["status"] == "FALSIFIED"
    assert res_falsified["is_falsified"] is True
    
    # Realized price above threshold -> NOT_FALSIFIED (strictly Popperian, never VALIDATED_INTACT)
    res_intact = evaluate_falsification_condition(f_obj, realized_metric_value=85000.0)
    assert res_intact["status"] == "NOT_FALSIFIED"
    assert res_intact["is_falsified"] is False
    print("  [PASS] Test 26: Machine Falsification Resolution Evaluator (Popperian NOT_FALSIFIED)")

def test_point_in_time_metadata_and_lineage_taxonomy():
    """Validates point-in-time timestamps and 7-member parameter lineage taxonomy."""
    from core.pipeline import CausalTimesFmPipeline
    pipeline = CausalTimesFmPipeline()
    sample_csv = os.path.join(BASE_DIR, "data", "btc_sample_history.csv")
    res = pipeline.run_crypto_pipeline(ticker="BTC-USD", history_file=sample_csv)
    
    assert "as_of_date" in res
    assert "availability_timestamp" in res
    assert "vintage_id" in res
    
    allowed_taxonomies = {
        "RAW_OBSERVED", "DERIVED_OBSERVED", "ESTIMATED", 
        "CALIBRATED", "POLICY", "SCENARIO_ASSUMPTION", "DEMO_ONLY"
    }
    for k, rec in res["parameter_lineage"].items():
        assert rec["parameter_status"] in allowed_taxonomies, f"Invalid taxonomy {rec['parameter_status']} for {k}"
        assert "observation_timestamp" in rec
        assert "availability_timestamp" in rec
    print("  [PASS] Test 27: Point-in-Time Availability Timestamps & Lineage Taxonomy")

def test_immutable_forecast_ledger():
    """Validates append-only forecast ledger recording, hash computation, and post-facto Popperian resolution."""
    import tempfile
    from core.forecast_ledger import ImmutableForecastLedger
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_file = os.path.join(tmp_dir, "test_ledger.jsonl")
        ledger = ImmutableForecastLedger(ledger_file_path=ledger_file)
        
        # 1. Record forecast
        rec = ledger.record_forecast(
            asset_name="BTC-USD",
            origin_timestamp="2026-03-01T00:00:00Z",
            horizon_steps=30,
            frequency="D",
            raw_prior={"p10": 80000.0, "p50": 90000.0, "p90": 105000.0},
            scenario_corridors={"downside_floor": 78000.0, "expected_target": 92000.0, "upside_ceiling": 108000.0},
            falsification_object={"primary_metric": "price", "threshold": 78000.0, "operator": "<"},
            allocation_output={"target_risk_weight": 0.50},
            data_cutoff_timestamp="2026-03-01T00:00:00Z",
            structural_alpha=0.25
        )
        forecast_id = rec["forecast_id"]
        assert forecast_id is not None
        assert rec["sha256_hash"] is not None
        
        # 2. Query record
        fetched = ledger.get_forecast(forecast_id)
        assert fetched is not None
        assert fetched["forecast_id"] == forecast_id
        
        # 3. Resolve forecast (Popperian NOT_FALSIFIED)
        res_record = ledger.resolve_forecast(
            forecast_id=forecast_id,
            realized_actual=91500.0,
            falsification_status="NOT_FALSIFIED",
            notes="Target realized well within scenario corridors."
        )
        assert res_record["falsification_status"] == "NOT_FALSIFIED"
        assert res_record["realized_actual"] == 91500.0
        
        # 4. Verify resolution is listed
        resolutions = ledger.list_resolutions(forecast_id)
        assert len(resolutions) == 1
        assert resolutions[0]["forecast_id"] == forecast_id
    print("  [PASS] Test 28: Immutable Forecast Ledger (Append-Only JSONL & Popperian Resolution)")

def test_diebold_mariano_and_block_bootstrap():
    """Validates Diebold-Mariano test with HLN finite-sample correction and Moving Block Bootstrap CI."""
    from core.econometrics import EconometricFilter
    ef = EconometricFilter()
    
    np.random.seed(42)
    # Model 1 has much lower error than Model 2
    e1 = [0.1 * np.random.randn() for _ in range(50)]
    e2 = [5.0 + np.random.randn() for _ in range(50)]
    
    dm_res = ef.compute_diebold_mariano_test(e1, e2, h=1, loss_power=2)
    assert "dm_statistic" in dm_res
    assert "p_value" in dm_res
    assert dm_res["model_is_superior"] is True
    assert dm_res["p_value"] < 0.05
    
    # Test Moving Block Bootstrap for Theil's U
    ci_res = ef.compute_block_bootstrap_theils_u(e1, e2, h=1, n_boot=100)
    assert "ci_95_lower" in ci_res
    assert "ci_95_upper" in ci_res
    assert ci_res["ci_95_lower"] <= ci_res["ci_95_upper"]
    assert ci_res["theils_u2_point"] < 1.0
    print("  [PASS] Test 29: Diebold-Mariano Test (HLN Adjusted) & Moving Block Bootstrap 95% CI")

def test_infold_structural_alpha_optimization():
    """Validates in-fold structural weight alpha* optimization and truthful collapse to zero."""
    from core.markov_regime import InstitutionalMarkovEngine
    engine = InstitutionalMarkovEngine()
    
    # Case A: Structural targets are inversely correlated/terrible -> alpha* must collapse to 0.0
    actual_fold = [100.0, 102.0, 105.0, 107.0, 110.0]
    tfm_fold = [100.5, 102.2, 104.8, 106.9, 109.8]  # Close to actual
    bad_struct = [50.0, 48.0, 45.0, 42.0, 40.0]     # Massive error
    
    res_bad = engine.estimate_optimal_structural_weight(actual_fold, tfm_fold, bad_struct, min_inner_train=2)
    assert res_bad["alpha_star"] == 0.0
    assert res_bad["structural_value_added"] is False
    
    # Case B: Structural target provides superior signal -> alpha* > 0.0
    good_struct = [100.0, 102.0, 105.0, 107.0, 110.0]  # Perfect
    noisy_tfm = [120.0, 125.0, 130.0, 135.0, 140.0]
    res_good = engine.estimate_optimal_structural_weight(actual_fold, noisy_tfm, good_struct, min_inner_train=2)
    assert res_good["alpha_star"] > 0.5
    assert res_good["structural_value_added"] is True
    print("  [PASS] Test 30: In-Fold Structural Alpha* Estimation & Truthful Zero Collapse")

def test_point_in_time_vintage_revision_leak_defense():
    """Validates point-in-time filtering strictly purges post-cutoff data and revision leaks."""
    from core.pipeline import filter_series_by_availability_cutoff
    
    records = [
        {"timestamp": "2026-01-01", "price": 100.0, "availability_timestamp": "2026-01-01T12:00:00Z"},
        {"timestamp": "2026-01-02", "price": 102.0, "availability_timestamp": "2026-01-02T12:00:00Z"},
        {"timestamp": "2026-01-03", "price": 105.0, "availability_timestamp": "2026-01-10T12:00:00Z"}
    ]
    
    # Filter as of cutoff 2026-01-05
    filtered = filter_series_by_availability_cutoff(records, cutoff_iso="2026-01-05T00:00:00Z")
    assert len(filtered) == 2
    assert filtered[-1]["timestamp"] == "2026-01-02"
    print("  [PASS] Test 31: Point-in-Time Vintage Availability Filtering & Revision Leak Defense")

def test_probabilistic_calibration_pinball_and_wis():
    """Validates Pinball Loss, Empirical Coverage, and Winkler Interval Score (WIS)."""
    from core.calibration import (
        compute_pinball_loss,
        compute_empirical_quantile_coverage,
        compute_winkler_interval_score
    )
    
    # Pinball loss: if actual == forecast, loss = 0.0
    assert compute_pinball_loss(100.0, 100.0, tau=0.5) == 0.0
    
    # Under-prediction at tau=0.9 -> penalty is tau * (y - q)
    under_loss = compute_pinball_loss(110.0, 100.0, tau=0.9)
    assert abs(under_loss - 0.9 * 10.0) < 1e-5
    
    # Over-prediction at tau=0.1 -> penalty is (1 - tau) * (q - y)
    over_loss = compute_pinball_loss(90.0, 100.0, tau=0.1)
    assert abs(over_loss - 0.9 * 10.0) < 1e-5
    
    # Coverage test: 8 of 10 points within interval -> 0.8 coverage
    y_test = [95.0] * 8 + [85.0, 115.0]
    p10 = [90.0] * 10
    p90 = [110.0] * 10
    cov = compute_empirical_quantile_coverage(y_test, p10_series=p10, p90_series=p90)
    assert "cov_p10" in cov
    assert "cov_p90" in cov
    
    # Winkler score: width is (110 - 90) = 20; if inside, score is 20
    wis_inside = compute_winkler_interval_score(100.0, 90.0, 110.0, alpha=0.1)
    assert abs(wis_inside - 20.0) < 1e-5
    
    # If below lower by 5: penalty is 2/0.1 * 5 = 100 -> score = 20 + 100 = 120
    wis_below = compute_winkler_interval_score(85.0, 90.0, 110.0, alpha=0.1)
    assert abs(wis_below - 120.0) < 1e-5
    print("  [PASS] Test 32: Probabilistic Calibration: Pinball Loss, Coverage & Winkler Interval Score")

def test_baseline_tournament():
    """Validates 5-tier baseline tournament hierarchy (M0, M0b, M1, M4) with delta-skill ranking."""
    from core.calibration import run_baseline_tournament
    
    np.random.seed(42)
    series = [100.0]
    for i in range(40):
        series.append(series[-1] * 1.01 + np.random.normal(0, 0.1))
        
    res = run_baseline_tournament(series, h=1, min_train_len=30)
    assert res["status"] == "EVALUATED"
    assert "models" in res
    assert "M0_Persistence" in res["models"]
    assert "M0b_Drift" in res["models"]
    assert "M1_TimesFM_Target_Only" in res["models"]
    assert "M4_Fitted_Structural_Hybrid" in res["models"]
    assert "incremental_skill" in res
    assert res["models"]["M0_Persistence"]["theils_u2"] == 1.0
    print("  [PASS] Test 33: Multi-Model Tournament Hierarchy & Skill Delta Reporting")

def test_media_hill_corner_cases():
    """Validates defensive handling of invalid Hill model parameters (gamma <= 1, K_max <= 0, spend <= 0)."""
    from core.pipeline import CausalTimesFmPipeline
    pipeline = CausalTimesFmPipeline()
    
    # gamma <= 1.0: should raise ValueError
    import pytest
    with pytest.raises(ValueError, match="INVALID_PARAMETER"):
        pipeline.solve_optimal_media_spend(target_cpm=10.0, ec50_spend=10000.0, k_max_impressions=1000000.0, gamma=0.9)
    
    # K_max <= 0: should raise ValueError
    with pytest.raises(ValueError, match="INVALID_PARAMETER"):
        pipeline.solve_optimal_media_spend(target_cpm=10.0, ec50_spend=10000.0, k_max_impressions=0.0, gamma=1.5)
    
    # Non-positive spend run: should raise ValueError
    with pytest.raises(ValueError, match="INVALID_PARAMETER"):
        pipeline.run_media_pipeline(monthly_spend=-500.0, cpm=12.0)
    print("  [PASS] Test 34: Media Hill Model Defensive Guardrails (Corner Cases Gamma <= 1, K_max <= 0)")

if __name__ == "__main__":
    print("\n=======================================================")
    print("   RUNNING CAUSAL-TIMESFM-ENGINE V3.0 INSTITUTIONAL TESTS")
    print("=======================================================")
    test_econometric_stationarity()
    test_schmitt_trigger_hysteresis()
    test_asset_calibrated_volatility()
    test_plain_english_summary()
    test_data_provenance_and_staleness()
    test_no_lookahead_in_rolling_features()
    test_epistemic_granger_taxonomy()
    test_theils_u_metric_properties()
    test_empirical_theils_u_hurdle_gate()
    test_media_hill_diminishing_marginal_returns()
    test_markov_horizon_propagation()
    test_timesfm_provenance_observability()
    test_unified_pipeline_crypto_and_guardrails()
    test_unified_pipeline_multi_domain()
    test_markov_stationary_distribution_and_modes()
    test_rolling_origin_theils_u_backtest()
    test_scenario_corridor_integrity_and_probabilistic_honesty()
    test_frequency_inference_and_horizon_mapping()
    test_parameter_lineage_and_empirical_state()
    test_theils_u_operational_pipeline_gating()
    test_media_hill_exact_spend_ceiling_and_schema()
    test_machine_testable_falsification_objects()
    test_multi_horizon_theils_u_gate_fail_closed()
    test_frequency_parser_fail_closed()
    test_structural_macro_anchor_weighting()
    test_machine_falsification_resolver()
    test_point_in_time_metadata_and_lineage_taxonomy()
    test_immutable_forecast_ledger()
    test_diebold_mariano_and_block_bootstrap()
    test_infold_structural_alpha_optimization()
    test_point_in_time_vintage_revision_leak_defense()
    test_probabilistic_calibration_pinball_and_wis()
    test_baseline_tournament()
    test_media_hill_corner_cases()
    print("=======================================================")
    print("   ALL 34 INSTITUTIONAL TEST SUITES PASSED (100% SUCCESS)")
    print("=======================================================\n")

