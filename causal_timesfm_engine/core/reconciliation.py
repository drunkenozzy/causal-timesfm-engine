"""
Reconciliation & Capital Allocation Engine
==========================================
Reconciles Engine 1 (TimesFM Statistical) and Engine 2 (Structural Causal):
  Delta = Structural Target - Statistical Forecast (TimesFM P50)

Enforces the 17-Point Capital Allocation Philosophy:
  Earned Cash Flow -> Speculative High-Beta (35%) -> Realize Gains -> 
  Durable Real Assets (50%) -> Liquidity Buffer (15%) -> Long-Term Optionality.
"""

class ReconciliationEngine:
    def __init__(self, delta_threshold_pct=15.0):
        self.delta_threshold = delta_threshold_pct

    def reconcile(self, statistical_output, structural_output):
        """
        Calculates divergence between statistical momentum and structural reality.
        """
        p50 = statistical_output["p50_expected"]
        struct_val = structural_output["structural_target"]
        current = statistical_output["current_price"]

        delta_dollar = struct_val - p50
        delta_pct = ((struct_val - p50) / current) * 100.0

        # Classify the divergence regime
        if delta_pct > self.delta_threshold:
            regime = "COILED_ASYMMETRIC_SPRING"
            interpretation = (
                "STRUCTURAL SURPLUS DETECTED: Fundamental adoption and clean tokenomics exceed "
                "pure chart momentum. High-probability asymmetric setup."
            )
            tactical_action = "ACCUMULATE systematically in controlled tranches. Upside convexity active."
            allocation_recommendation = "Maintain speculative allocation up to 35% max threshold."
            
        elif delta_pct < -self.delta_threshold:
            regime = "OVEREXTENDED_SPECULATIVE_BUBBLE"
            interpretation = (
                "MOMENTUM / REALIZATION CRISIS RISK: Market price is riding speculative momentum "
                "or leveraged derivatives positioning disconnected from structural cash-flow or float dilution."
            )
            tactical_action = "HARVEST PROFITS aggressively. Do not add capital; execute laddered exits into USDT."
            allocation_recommendation = "Rotate 70-80% of gains into Durable Real Assets (50%) and Liquidity Buffer (15%)."
            
        else:
            regime = "CONGRUENT_CYCLE_EXPANSION"
            interpretation = (
                "MECHANICALLY SUPPORTED MOMENTUM: Chart trajectory and structural macro drivers "
                "are in equilibrium."
            )
            tactical_action = "HOLD positions with trailing P10 stop-loss. Respect the TimesFM corridor."
            allocation_recommendation = "Maintain current allocations; rebalance on boundary breaches."

        return {
            "current_price": current,
            "timesfm_p10_floor": statistical_output["p10_downside"],
            "timesfm_p50_baseline": p50,
            "timesfm_p90_ceiling": statistical_output["p90_upside"],
            "structural_target": struct_val,
            "delta_dollar": round(delta_dollar, 4),
            "delta_pct": round(delta_pct, 2),
            "regime": regime,
            "interpretation": interpretation,
            "tactical_action": tactical_action,
            "allocation_recommendation": allocation_recommendation,
            "falsifiability_check": (
                f"Thesis falsified if price closes below P10 support (${statistical_output['p10_downside']}) "
                f"or if Minsky regime degrades into Ponzi insolvency."
            )
        }
