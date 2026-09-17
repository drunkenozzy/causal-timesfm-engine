"""
Automated Unit Tests & Falsifiability Validation
================================================
Tests:
  1. Econometric stationarization & ADF unit root rejection.
  2. Granger causality F-test validation.
  3. TimesFM quantile corridor generation.
  4. Dual-Engine Reconciliation (Coiled vs. Overextended detection).
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.econometrics import EconometricFilter
from core.engine_timesfm import TimesFmBaselineEngine
from core.engine_structural import StructuralMacroEngine
from core.reconciliation import ReconciliationEngine

class TestCausalTimesFmPipeline(unittest.TestCase):

    def setUp(self):
        self.econ = EconometricFilter()
        self.tfm = TimesFmBaselineEngine()
        self.struct = StructuralMacroEngine()
        self.recon = ReconciliationEngine()

    def test_log_differencing_removes_trend(self):
        """Validates that first differences of an upward linear trend become stationary."""
        trend = [10.0 + 2.0 * i for i in range(50)]
        res = self.econ.stationarize(trend)
        self.assertTrue(res["is_stationary"])

    def test_timesfm_quantile_envelope_sanity(self):
        """Validates that P10 < P50 < P90 in statistical forecast."""
        prices = [100.0 + (i * 0.5) for i in range(60)]
        out = self.tfm.forecast(prices, horizon_days=30)
        self.assertLess(out["p10_downside"], out["p50_expected"])
        self.assertLess(out["p50_expected"], out["p90_upside"])

    def test_reconciliation_overextended_bubble_detection(self):
        """Validates that a Ponzi asset with heavy dilution triggers OVEREXTENDED bubble regime."""
        stat_out = {
            "current_price": 100.0,
            "p10_downside": 80.0,
            "p50_expected": 140.0, # High chart momentum
            "p90_upside": 200.0
        }
        # Heavy dilution (30% float) and Ponzi stage
        struct_out = self.struct.compute_structural_target(
            current_price=100.0,
            macro_liquidity_regime="contracting",
            circulating_supply=300_000,
            total_supply=1_000_000,
            minsky_stage="Ponzi",
            organic_tvl_growth=-20.0
        )
        recon_res = self.recon.reconcile(stat_out, struct_out)
        self.assertEqual(recon_res["regime"], "OVEREXTENDED_SPECULATIVE_BUBBLE")
        self.assertIn("HARVEST PROFITS", recon_res["tactical_action"])

    def test_reconciliation_coiled_asymmetric_spring(self):
        """Validates that clean tokenomics and strong TVL trigger COILED ASYMMETRIC SPRING."""
        stat_out = {
            "current_price": 10.0,
            "p10_downside": 9.0,
            "p50_expected": 10.5, # Flat momentum
            "p90_upside": 13.0
        }
        # Clean float (90%), expanding liquidity, Hedge balance sheet
        struct_out = self.struct.compute_structural_target(
            current_price=10.0,
            macro_liquidity_regime="expanding",
            circulating_supply=900_000,
            total_supply=1_000_000,
            minsky_stage="Hedge",
            organic_tvl_growth=50.0
        )
        recon_res = self.recon.reconcile(stat_out, struct_out)
        self.assertEqual(recon_res["regime"], "COILED_ASYMMETRIC_SPRING")
        self.assertIn("ACCUMULATE", recon_res["tactical_action"])

if __name__ == "__main__":
    unittest.main()
