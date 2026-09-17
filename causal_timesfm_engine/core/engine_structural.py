"""
Engine 2: Causal Structural Macro & Political-Economy Engine
===========================================================
Models the real-world mechanical constraints:
  1. Global Liquidity Impulses (Central Bank Balance Sheets & US Real Yields).
  2. Minsky Financial Fragility Classifier (Hedge -> Speculative -> Ponzi).
  3. Circuits of Capital & Rent Extraction (M -> C -> M' realized cash flow).
  4. Tokenomics & Supply Dilution Drag (Circulating Float vs. Unlocks).
"""

class StructuralMacroEngine:
    def __init__(self):
        pass

    def classify_minsky_regime(self, revenue_to_debt_ratio, leverage_ratio, has_organic_cashflow=True):
        """
        Classifies financial structure into Minsky's 3 stages of financial instability:
          - Hedge: Operating cash flows cover all interest and debt obligations.
          - Speculative: Cash flows cover interest payments, but debt must be continually rolled over.
          - Ponzi: Cash flows cover neither interest nor principal; reliance entirely on rising asset prices.
        """
        if not has_organic_cashflow or revenue_to_debt_ratio < 0.2:
            return {
                "stage": "Ponzi",
                "risk_multiplier": 0.50,
                "description": "Reliance solely on new capital inflows / asset price appreciation. Extreme fragility to liquidity contraction."
            }
        elif revenue_to_debt_ratio < 1.0 or leverage_ratio > 4.0:
            return {
                "stage": "Speculative",
                "risk_multiplier": 0.85,
                "description": "Cash flows cover current obligations but cannot absorb refinancing shocks or macro rate spikes."
            }
        else:
            return {
                "stage": "Hedge",
                "risk_multiplier": 1.15,
                "description": "Robust balance sheet; organic income exceeds commitments. Highly resilient through economic cycles."
            }

    def evaluate_dilution_drag(self, circulating_supply, total_supply):
        """
        Evaluates structural tokenomics dilution.
        Circulating Float Ratio = Circulating Supply / Total Supply (or FDV).
        """
        if total_supply <= 0:
            return {"circ_ratio": 1.0, "dilution_drag": 0.0, "severity": "Unknown"}

        circ_ratio = min(1.0, circulating_supply / total_supply)

        if circ_ratio >= 0.70:
            drag = 0.05   # Low drag: over 70% already in market
            severity = "Low (Float Clean)"
        elif circ_ratio >= 0.45:
            drag = 0.25   # Moderate drag
            severity = "Moderate (Upcoming vesting pressure)"
        else:
            drag = 0.55   # Severe drag: heavy unlocks will dilute price (e.g. ARB / XAI)
            severity = "Severe Dilution (Substantial unvested overhang)"

        return {
            "circulating_ratio": round(circ_ratio, 3),
            "dilution_drag": round(drag, 2),
            "severity": severity
        }

    def compute_structural_target(self, current_price, macro_liquidity_regime="expanding", 
                                  circulating_supply=1.0, total_supply=1.0, 
                                  minsky_stage="Hedge", organic_tvl_growth=0.0):
        """
        Synthesizes causal macro variables into a Structural Valuation Anchor:
          Structural Multiplier = (Liquidity Impulses) * (Minsky Multiplier) * (1 - Dilution Drag) * (1 + TVL Growth)
        """
        # 1. Macro Liquidity Component
        liquidity_multipliers = {
            "expanding": 1.25,     # Central banks easing / M2 expanding
            "neutral": 1.00,       # Rangebound liquidity
            "contracting": 0.80    # QT / Real yields rising
        }
        liq_mult = liquidity_multipliers.get(macro_liquidity_regime, 1.00)

        # 2. Minsky Multiplier
        minsky_info = self.classify_minsky_regime(
            revenue_to_debt_ratio=1.5 if minsky_stage == "Hedge" else (0.5 if minsky_stage == "Speculative" else 0.0),
            leverage_ratio=1.0 if minsky_stage == "Hedge" else 5.0,
            has_organic_cashflow=(minsky_stage != "Ponzi")
        )
        minsky_mult = minsky_info["risk_multiplier"]

        # 3. Dilution Component
        dilution_info = self.evaluate_dilution_drag(circulating_supply, total_supply)
        dilution_factor = 1.0 - dilution_info["dilution_drag"]

        # 4. Organic Adoption / TVL Component
        growth_factor = 1.0 + max(-0.5, min(1.5, organic_tvl_growth / 100.0))

        # Composite Structural Value Target
        structural_multiplier = liq_mult * minsky_mult * dilution_factor * growth_factor
        structural_target = current_price * structural_multiplier

        return {
            "current_price": current_price,
            "structural_target": round(structural_target, 4),
            "structural_multiplier": round(structural_multiplier, 3),
            "structural_return_pct": round((structural_multiplier - 1.0) * 100, 2),
            "minsky_stage": minsky_info["stage"],
            "minsky_description": minsky_info["description"],
            "dilution_severity": dilution_info["severity"],
            "circulating_ratio": dilution_info["circulating_ratio"]
        }
