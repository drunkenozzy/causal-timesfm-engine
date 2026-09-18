"""
Institutional Reconciliation & Rule 6 Capital Allocation Engine v2.0
=====================================================================
Reconciles Foundation Model statistical trajectories with Causal Structural Reality.
Enforces Rule 6 of the 17-Point Framework:
  Earned Cash Flow -> Speculative High-Beta (35% Max) -> Realize Gains ->
  Durable Real Assets (50%) -> Liquidity Buffer (15%) -> Long-Term Optionality.

Includes the Plain-English Executive Summary Generator.
"""

class ReconciliationEngine:
    def __init__(self, delta_threshold_pct=15.0):
        self.delta_threshold = delta_threshold_pct

    def compute_allocation_weights(self, markov_output, decoupling_active=False, momentum_positive=True, tax_drag_threshold=0.15):
        """
        Computes dynamic Rule 6 asset allocation weights using Schmitt Trigger Hysteresis.
        """
        in_ponzi = markov_output.get("in_ponzi_regime", False)
        xi_state = markov_output.get("state_vector", [0.8, 0.15, 0.05])
        p_spec = float(xi_state[1])
        p_ponzi = float(xi_state[2])

        if in_ponzi or p_ponzi > 0.40:
            target_risk_weight = 0.15  # Rule 6 Risk-Off Floor
            cash_buffer_weight = 0.85  # Rotate 85% into Liquidity Buffer / Real Assets
            regime = "PONZI_LIQUIDATION_CRUNCH"
            tactical_action = "CAPITAL PRESERVATION: Rotate 85% into cash buffer / durable real assets. Cut high-beta exposure."
            
        elif p_spec > 0.50:
            target_risk_weight = 0.35  # Rule 6 Speculative Cap (Strictly max 35%)
            cash_buffer_weight = 0.65
            regime = "SPECULATIVE_OVEREXTENSION"
            tactical_action = "HARVEST PROFITS: Cap speculative beta at 35%. Ladder out profits into durable real assets."

        elif decoupling_active and momentum_positive:
            target_risk_weight = 0.85  # Full Prudent Allocation (No cash drag)
            cash_buffer_weight = 0.15
            regime = "ORGANIC_DECOUPLING_EXPANSION"
            tactical_action = "AGGRESSIVE ACCUMULATION: On-chain money creation is accelerating. Allocate up to 85%."

        elif xi_state[0] > 0.50 and momentum_positive:
            target_risk_weight = 0.70
            cash_buffer_weight = 0.30
            regime = "CONGRUENT_HEDGE_EXPANSION"
            tactical_action = "MAINTAIN EXPOSURE: Systemic conditions balanced. Trail stops at TimesFM P10 floor."

        else:
            target_risk_weight = 0.40
            cash_buffer_weight = 0.60
            regime = "TRANSITIONAL_DEFENSIVE"
            tactical_action = "DEFENSIVE POSTURE: Rebalance toward 60% cash buffer until clear breakout confirms."

        return {
            "target_risk_weight": round(target_risk_weight, 2),
            "cash_buffer_weight": round(cash_buffer_weight, 2),
            "regime": regime,
            "tactical_action": tactical_action,
            "ponzi_probability": round(p_ponzi, 4),
            "in_ponzi_regime": in_ponzi
        }

    def generate_plain_english_summary(self, asset_name, current_price, currency_symbol, 
                                        forecast_output, allocation_output, falsifiability_condition):
        """
        Generates a transparent, jargon-free Plain-English Executive Summary
        designed for real-world capital decisions.
        """
        regime = allocation_output["regime"]
        risk_w = int(allocation_output["target_risk_weight"] * 100)
        cash_w = int(allocation_output["cash_buffer_weight"] * 100)
        p10 = forecast_output["reconciled_p10"]
        p50 = forecast_output["reconciled_p50"]
        p90 = forecast_output["reconciled_p90"]

        # Plain English regime translation
        regime_translations = {
            "PONZI_LIQUIDATION_CRUNCH": "LIQUIDITY CRUNCH / DANGER ZONE (Debt or money supply is draining; high risk of cascading selloffs)",
            "SPECULATIVE_OVEREXTENSION": "LATE-CYCLE EUPHORIA (Market is running hot on high leverage; upside is limited, downside risk is growing)",
            "ORGANIC_DECOUPLING_EXPANSION": "HEALTHY ACCUMULATION / EXPANSION (Real on-chain capital is entering the market; strong economic tailwinds)",
            "CONGRUENT_HEDGE_EXPANSION": "STEADY / STABLE GROWTH (Fundamentals and price momentum are in healthy balance)",
            "TRANSITIONAL_DEFENSIVE": "CHOPPY / UNCERTAIN (No strong trend; market is consolidating)"
        }
        plain_regime = regime_translations.get(regime, regime)

        summary = f"""
================================================================================
                    PLAIN-ENGLISH EXECUTIVE DECISION SUMMARY
================================================================================
Asset Evaluated: {asset_name} (Current Price: {currency_symbol}{current_price:,.2f})

1. WHERE WE STAND TODAY:
   Current Market Phase: {plain_regime}
   Systemic Danger Level: {allocation_output['ponzi_probability']*100:.1f}% risk of liquidity cascade.

2. WHAT TO DO WITH YOUR MONEY (RULE 6 ALLOCATION):
   Recommended Position: {risk_w}% Invested in {asset_name} | {cash_w}% in Cash Buffer / Real Assets.
   Tactical Directive:   {allocation_output['tactical_action']}

3. YOUR VALUATION & RISK NUMBERS (EXPECTED HORIZON):
   - Worst-Case Downside Floor (P10):   {currency_symbol}{p10:,.2f} ({((p10/current_price)-1)*100:+.1f}%)
   - Most Likely Expected Value (P50):  {currency_symbol}{p50:,.2f} ({((p50/current_price)-1)*100:+.1f}%)
   - Best-Case Ceiling (P90):          {currency_symbol}{p90:,.2f} ({((p90/current_price)-1)*100:+.1f}%)

4. WHAT WOULD PROVE THIS ANALYSIS WRONG (FALSIFIABILITY):
   {falsifiability_condition}
================================================================================
"""
        return summary.strip()
