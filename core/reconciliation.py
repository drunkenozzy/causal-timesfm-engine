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

    def reconcile(self, statistical_output, structural_output):
        """
        Dual-Engine Reconciliation:
        Compares the statistical trajectory against the structural valuation anchor.
        Evaluates whether chart momentum is supported by balance sheet reality.
        """
        curr_p = statistical_output.get("current_price", 100.0)
        p50 = statistical_output.get("p50_expected", curr_p)
        p10 = statistical_output.get("p10_downside", curr_p * 0.9)
        p90 = statistical_output.get("p90_upside", curr_p * 1.2)
        
        struct_target = structural_output.get("structural_target", curr_p)
        minsky_stage = structural_output.get("minsky_stage", "Hedge")
        
        delta_pct = ((p50 - struct_target) / struct_target) * 100.0 if struct_target > 0 else 0.0
        
        # Detect structural divergence
        if minsky_stage == "Ponzi" or delta_pct > self.delta_threshold:
            regime = "OVEREXTENDED_SPECULATIVE_BUBBLE"
            tactical_action = "HARVEST PROFITS: Structural floor is far below momentum expectation. Reduce high-beta exposure."
            reconciled_target = min(p50, struct_target)
        elif minsky_stage == "Hedge" and ((struct_target - p50) / p50 * 100.0) > self.delta_threshold:
            regime = "COILED_ASYMMETRIC_SPRING"
            tactical_action = "ACCUMULATE: Strong fundamentals and clean balance sheet create asymmetric upside."
            reconciled_target = max(p50, struct_target)
        else:
            regime = "CONGRUENT_EXPANSION"
            tactical_action = "MAINTAIN EXPOSURE: Statistical momentum and structural reality are aligned."
            reconciled_target = p50

        return {
            "regime": regime,
            "tactical_action": tactical_action,
            "reconciled_target": round(reconciled_target, 2),
            "delta_pct": round(delta_pct, 2),
            "reconciled_p10": round(min(p10, struct_target * 0.85), 2),
            "reconciled_p50": round(reconciled_target, 2),
            "reconciled_p90": round(max(p90, struct_target * 1.15), 2),
            "structural_anchor": round(struct_target, 2),
            "statistical_p50": round(p50, 2)
        }

    def compute_allocation_weights(self, markov_output, decoupling_active=False, momentum_positive=True, tax_drag_threshold=0.15, is_synthetic_mode=False):
        """
        Computes dynamic Rule 6 asset allocation weights using Schmitt Trigger Hysteresis.
        NOTE: This expresses a normative risk policy (Rule 6), not an empirical discovery of TimesFM.
        If is_synthetic_mode is True, allocation execution is hard-suppressed to prevent capital deployment on synthetic data.
        """
        in_ponzi = markov_output.get("in_ponzi_regime", False)
        xi_state = markov_output.get("state_vector", [0.8, 0.15, 0.05])
        p_spec = float(xi_state[1])
        p_ponzi = float(xi_state[2])
        fragility_score = markov_output.get("fragility_score", round(p_ponzi * 100, 1))

        if is_synthetic_mode:
            return {
                "target_risk_weight": 0.0,
                "cash_buffer_weight": 0.0,
                "regime": "SYNTHETIC_BENCHMARK_DEMO",
                "tactical_action": "[ALLOCATION SUPPRESSED: SYNTHETIC DEMONSTRATION ONLY - CANNOT MAKE REAL CAPITAL ALLOCATIONS WITHOUT EMPIRICAL TIME-SERIES DATA]",
                "fragility_score": fragility_score,
                "ponzi_probability": round(p_ponzi, 4),
                "in_ponzi_regime": in_ponzi,
                "allocation_disabled": True,
                "policy_type": "NORMATIVE_RULE_6_RISK_POLICY (SUPPRESSED)"
            }

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
            tactical_action = "MAINTAIN EXPOSURE: Systemic conditions balanced. Trail stops at Downside Scenario Floor."

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
            "fragility_score": fragility_score,
            "ponzi_probability": round(p_ponzi, 4),  # Preserved for backward compatibility
            "in_ponzi_regime": in_ponzi,
            "allocation_disabled": False,
            "policy_type": "NORMATIVE_RULE_6_RISK_POLICY"
        }

    def generate_plain_english_summary(self, asset_name, current_price, currency_symbol, 
                                        forecast_output, allocation_output, falsifiability_condition,
                                        raw_prior=None):
        """
        Generates a transparent, jargon-free Plain-English Executive Summary
        designed for real-world capital decisions.
        Cleanly separates:
          1. Empirical Model Findings
          2. Normative Decision Policy (Rule 6)
          3. Valuation & Risk Numbers (Foundation Prior vs Scenario Envelope)
          4. Observable Falsifiability Condition
        """
        regime = allocation_output["regime"]
        alloc_disabled = allocation_output.get("allocation_disabled", False)
        risk_w = int(allocation_output.get("target_risk_weight", 0.0) * 100)
        cash_w = int(allocation_output.get("cash_buffer_weight", 0.0) * 100)
        
        floor = forecast_output.get("downside_floor", forecast_output.get("reconciled_p10", current_price * 0.9))
        target = forecast_output.get("expected_target", forecast_output.get("reconciled_p50", current_price))
        ceiling = forecast_output.get("upside_ceiling", forecast_output.get("reconciled_p90", current_price * 1.1))
        
        frag_score = allocation_output.get("fragility_score", round(allocation_output.get("ponzi_probability", 0.0) * 100, 1))

        regime_translations = {
            "PONZI_LIQUIDATION_CRUNCH": "LIQUIDITY CRUNCH / ELEVATED FRAGILITY (Net money supply contraction; high sensitivity to selloffs)",
            "SPECULATIVE_OVEREXTENSION": "LATE-CYCLE SPECULATION (Elevated leverage / momentum; upside constrained relative to tail risk)",
            "ORGANIC_DECOUPLING_EXPANSION": "ORGANIC EXPANSION (Net on-chain base money creation; macro tailwinds)",
            "CONGRUENT_HEDGE_EXPANSION": "BALANCED EXPANSION (Fundamentals and price momentum in healthy alignment)",
            "TRANSITIONAL_DEFENSIVE": "CONSOLIDATION / DEFENSIVE (Directional trend unconfirmed; rangebound)",
            "SYNTHETIC_BENCHMARK_DEMO": "SYNTHETIC DEMONSTRATION MODE (No empirical time-series data supplied)"
        }
        plain_regime = regime_translations.get(regime, regime)

        # Prior display if available
        prior_section = ""
        if raw_prior:
            prior_section = f"""
   [FORECAST PROVENANCE: FOUNDATION PRIOR vs. MECHANISM-AWARE SCENARIOS]
   - TimesFM Foundation Model Quantile Prior:
       P10 (10th percentile): {currency_symbol}{raw_prior.get('p10_downside', floor):,.2f}
       P50 (Median expected): {currency_symbol}{raw_prior.get('p50_expected', target):,.2f}
       P90 (90th percentile): {currency_symbol}{raw_prior.get('p90_upside', ceiling):,.2f}
   - Mechanism-Aware Conditioned Scenario Corridors:
       Downside Scenario Floor:       {currency_symbol}{floor:,.2f} ({((floor/current_price)-1)*100:+.1f}%)
       Mechanism-Adjusted Central Target: {currency_symbol}{target:,.2f} ({((target/current_price)-1)*100:+.1f}%)
       Upside Scenario Ceiling:       {currency_symbol}{ceiling:,.2f} ({((ceiling/current_price)-1)*100:+.1f}%)
   [Note: Scenario corridors represent structural stress bounds, not linear mixture quantiles (Q_sum != sum Q).]
"""

        # Section 2 Allocation Display
        if alloc_disabled:
            section_2 = f"""2. WHAT TO DO WITH YOUR MONEY (RULE 6 CAPITAL ALLOCATION):
   [ALLOCATION DISABLED: SYNTHETIC DEMONSTRATION CONE ACTIVE]
   Real capital allocation is disabled when running on synthetic benchmark cones.
   Supply empirical historical prices via --history-file or live on-chain feeds to activate Rule 6 capital execution.
   - Status Directive:  {allocation_output['tactical_action']}"""
        else:
            section_2 = f"""2. WHAT TO DO WITH YOUR MONEY (RULE 6 CAPITAL ALLOCATION):
   [Note: Investor risk framework rule, distinct from empirical model forecasts.]
   - Position Target:   {risk_w}% Invested in {asset_name} | {cash_w}% in Cash Buffer / Real Assets.
   - Tactical Action:   {allocation_output['tactical_action']}"""

        summary = f"""
================================================================================
                    PLAIN-ENGLISH EXECUTIVE DECISION SUMMARY
================================================================================
Asset Evaluated: {asset_name} (Current Price: {currency_symbol}{current_price:,.2f})

1. WHERE WE STAND TODAY (EMPIRICAL MODEL FINDINGS):
   - Evaluated Regime State: {plain_regime}
   - Systemic Fragility Score: {frag_score:.1f} / 100
     [Note: State posterior score from dynamic TVTP filter; not an empirical frequency probability.]

{section_2}

3. VALUATION & SCENARIO CORRIDORS (EXPECTED HORIZON):{prior_section if prior_section else f'''
   - Downside Scenario Floor:       {currency_symbol}{floor:,.2f} ({((floor/current_price)-1)*100:+.1f}%)
   - Mechanism-Adjusted Central Target: {currency_symbol}{target:,.2f} ({((target/current_price)-1)*100:+.1f}%)
   - Upside Scenario Ceiling:       {currency_symbol}{ceiling:,.2f} ({((ceiling/current_price)-1)*100:+.1f}%)
   [Note: Scenario corridors represent structural stress bounds, not linear mixture quantiles (Q_sum != sum Q).]'''}

4. WHAT WOULD PROVE THIS ANALYSIS WRONG (FALSIFIABILITY):
   {falsifiability_condition}
================================================================================
"""
        return summary.strip()
