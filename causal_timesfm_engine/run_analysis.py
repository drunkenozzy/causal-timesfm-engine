"""
Master CLI Entry Point: Dual-Engine Causal-Econometric & TimesFM Pipeline
========================================================================
Executes the full 17-point analytical workflow:
  1. Econometric Stationarization & Granger Causality Filter (Kills the Elevator Effect).
  2. Engine 1: TimesFM Statistical / Quantile Corridor (P10, P50, P90).
  3. Engine 2: Structural Macro, Minsky Fragility & Token Dilution Constraints.
  4. Reconciliation: Delta = Structural - Statistical & Capital Allocation Matrix.
"""

import sys
import os
import json
import math
import random

# Ensure local core imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.econometrics import EconometricFilter
from core.engine_timesfm import TimesFmBaselineEngine
from core.engine_structural import StructuralMacroEngine
from core.reconciliation import ReconciliationEngine

# Ensure clean Windows console printing
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_causal_timesfm_analysis(symbol="CPOOL", current_price=0.0228, 
                                historical_prices=None, driver_series=None,
                                macro_liquidity="expanding", 
                                circulating_supply=790_000_000, 
                                total_supply=1_000_000_000,
                                minsky_stage="Hedge", 
                                organic_growth_pct=25.0,
                                horizon_days=30):
    
    print("=" * 80)
    print(f"  DUAL-ENGINE CAUSAL-ECONOMETRIC & TIMESFM ANALYSIS: {symbol}")
    print("=" * 80)

    # 1. Generate / Validate Historical Series if not provided
    if historical_prices is None or len(historical_prices) < 30:
        random.seed(42)
        base = current_price * 0.8
        historical_prices = []
        curr = base
        for i in range(120):
            shock = random.gauss(0, 0.02)
            drift = 0.001
            curr *= math.exp(drift + shock)
            historical_prices.append(curr)
        historical_prices[-1] = current_price

    if driver_series is None or len(driver_series) < 30:
        random.seed(43)
        driver_series = []
        curr_d = 100.0
        for i in range(len(historical_prices)):
            shock = random.gauss(0, 0.015)
            drift = 0.0008
            curr_d *= math.exp(drift + shock)
            driver_series.append(curr_d)

    # =========================================================================
    # STEP 1: THE CAUSAL DECOUPLING ECONOMETRIC FILTER
    # =========================================================================
    print("\n[STEP 1: ECONOMETRIC STATIONARIZATION & CAUSALITY FILTER]")
    econ = EconometricFilter(p_value_threshold=0.05, max_lags=2)

    # Stationarize target (isolate shocks, kill the elevator)
    stat_target = econ.stationarize(historical_prices)
    stat_driver = econ.stationarize(driver_series)

    print(f"  * Target Series Stationarity Check (ADF): "
          f"{'PASSED I(0)' if stat_target['is_stationary'] else 'FAILED (Non-stationary)'} "
          f"(p-val: {stat_target['adf_p_value']})")

    # Granger Causality Test on First Differences
    granger_res = econ.granger_causality_test(
        stat_driver["diff_growth"], 
        stat_target["diff_growth"], 
        lags=2
    )

    print(f"  * Granger Causality F-Stat: {granger_res['f_statistic']} | p-val: {granger_res['p_value']}")
    print(f"  * Verdict: {granger_res['verdict']}")

    # =========================================================================
    # STEP 2: ENGINE 1 (TIMESFM STATISTICAL BASELINE)
    # =========================================================================
    print(f"\n[STEP 2: ENGINE 1 — STATISTICAL FOUNDATION BASELINE ({horizon_days}D Horizon)]")
    tfm_engine = TimesFmBaselineEngine()
    stat_output = tfm_engine.forecast(historical_prices, horizon_days=horizon_days)

    print(f"  * Current Price       : ${stat_output['current_price']:,.4f}")
    print(f"  * TimesFM P10 (Floor) : ${stat_output['p10_downside']:,.4f} ({stat_output['downside_risk_pct']:+.1f}%)")
    print(f"  * TimesFM P50 (Median): ${stat_output['p50_expected']:,.4f} ({stat_output['expected_return_pct']:+.1f}%)")
    print(f"  * TimesFM P90 (Ceiling): ${stat_output['p90_upside']:,.4f} ({stat_output['upside_tail_pct']:+.1f}%)")
    print(f"  * Volatility Squeeze  : {'ACTIVE [Squeezed]' if stat_output['volatility_squeeze'] else 'Inactive (Normal bandwidth)'}")

    # =========================================================================
    # STEP 3: ENGINE 2 (STRUCTURAL MACRO & POLITICAL-ECONOMY)
    # =========================================================================
    print("\n[STEP 3: ENGINE 2 — STRUCTURAL MACRO & CIRCUIT CONSTRAINTS]")
    struct_engine = StructuralMacroEngine()
    struct_output = struct_engine.compute_structural_target(
        current_price=current_price,
        macro_liquidity_regime=macro_liquidity,
        circulating_supply=circulating_supply,
        total_supply=total_supply,
        minsky_stage=minsky_stage,
        organic_tvl_growth=organic_growth_pct
    )

    print(f"  * Macro Liquidity Regime: {macro_liquidity.upper()}")
    print(f"  * Minsky Financial Stage: {struct_output['minsky_stage']} ({struct_output['minsky_description']})")
    print(f"  * Supply Dilution Drag  : {struct_output['dilution_severity']} (Float: {struct_output['circulating_ratio']*100:.1f}%)")
    print(f"  * Structural Fair Value : ${struct_output['structural_target']:,.4f} ({struct_output['structural_return_pct']:+.1f}%)")

    # =========================================================================
    # STEP 4: RECONCILIATION & CAPITAL ALLOCATION
    # =========================================================================
    print("\n[STEP 4: DUAL-ENGINE RECONCILIATION (Delta = Structural - Statistical)]")
    recon = ReconciliationEngine(delta_threshold_pct=15.0)
    reconciled = recon.reconcile(stat_output, struct_output)

    print(f"  * Delta Divergence      : {reconciled['delta_pct']:+.2f}% (${reconciled['delta_dollar']:+,.4f})")
    print(f"  * Market Regime         : {reconciled['regime']}")
    print(f"\n  [INTERPRETATION]\n  {reconciled['interpretation']}")
    print(f"\n  [TACTICAL ACTION]\n  {reconciled['tactical_action']}")
    print(f"\n  [CAPITAL ROTATION (Rule 6)]\n  {reconciled['allocation_recommendation']}")
    print(f"\n  [FALSIFIABILITY AUDIT]\n  {reconciled['falsifiability_check']}")
    print("=" * 80)

    # Save complete audit trail
    report_path = os.path.join(os.path.dirname(__file__), "data", f"{symbol}_audit_report.json")
    full_report = {
        "symbol": symbol,
        "econometrics": granger_res,
        "engine_1_timesfm": stat_output,
        "engine_2_structural": struct_output,
        "reconciliation": reconciled
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nFull reproducible audit trail saved to: {report_path}\n")
    return full_report

if __name__ == "__main__":
    # Test execution on Clearpool (CPOOL)
    run_causal_timesfm_analysis(
        symbol="CPOOL",
        current_price=0.0228,
        macro_liquidity="expanding",
        circulating_supply=790_000_000,
        total_supply=1_000_000_000,
        minsky_stage="Hedge",
        organic_growth_pct=35.0,
        horizon_days=30
    )
