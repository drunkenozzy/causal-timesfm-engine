"""
Causal TimesFM Engine v2.2: Unified Analysis Pipeline
=====================================================
Single source of truth for both CLI (run_analysis.py) and Streamlit Web UI (app.py).
Centralizes the 6-stage forecasting & capital governance pipeline:
  1. Data Ingestion & Provenance (Empirical Series vs Synthetic Demo Cone)
  2. Foundation Model Statistical Prior (TimesFmBaselineEngine with degraded-mode telemetry)
  3. TVTP Markov Regime Filtering & Forward Horizon Propagation
  4. Mechanism-Aware Scenario Corridor Envelope (Floor / Central / Ceiling)
  5. Normative Rule 6 Capital Allocation (Hard-suppressed during Synthetic Demos)
  6. Plain-English Executive Decision Card Generation
"""

import os
import math
import json
from datetime import datetime, timezone

from core.markov_regime import InstitutionalMarkovEngine
from core.onchain_liquidity import OnChainLiquidityEngine
from core.reconciliation import ReconciliationEngine
from core.engine_timesfm import TimesFmBaselineEngine

def generate_synthetic_history(current_val, days=60, daily_vol=0.04, daily_drift=0.001):
    """
    Constructs a reproducible synthetic history cone ending at current_val
    matching target volatility and drift when empirical series is absent.
    """
    history = [current_val]
    val = current_val
    for i in range(1, days):
        shock = math.sin(i * 0.7) * daily_vol - daily_drift
        val = val / (1.0 + shock)
        history.insert(0, max(val, 0.01))
    return history

def load_history_series(file_path):
    """Loads empirical historical prices from a CSV or text file."""
    vals = []
    if file_path and os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.lower().startswith(("date", "time", "timestamp", "price")):
                    parts = line.split(",")
                    try:
                        v = float(parts[-1].strip() if len(parts) > 1 else parts[0].strip())
                        vals.append(v)
                    except ValueError:
                        continue
    return vals

class CausalTimesFmPipeline:
    """
    Unified institutional analysis pipeline orchestrating data provenance,
    foundation model inference, causal macro conditioning, and risk policy.
    """
    def __init__(self, delta_threshold_pct=15.0):
        self.liq_engine = OnChainLiquidityEngine()
        self.reconciler = ReconciliationEngine(delta_threshold_pct=delta_threshold_pct)
        self.tfm_engine = TimesFmBaselineEngine()

    def run_crypto_pipeline(self, ticker="BTC-USD", current_price=94000.0, history_file=None, history_series=None, horizon_days=30):
        markov = InstitutionalMarkovEngine(asset_daily_std=0.045)
        
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        mcap_prov = self.liq_engine.get_stablecoin_mcap_with_provenance(today_str)
        mcap = mcap_prov["mcap"]
        decoupling = (mcap is not None and mcap > 150e9)

        # 1. Historical Lineage Determination
        if history_series and len(history_series) > 0:
            history = list(history_series)
            current_price = history[-1]
            data_mode = "EMPIRICAL_HISTORICAL_DATA"
        elif history_file and os.path.exists(history_file):
            emp = load_history_series(history_file)
            if emp:
                history = emp
                current_price = history[-1]
                data_mode = "EMPIRICAL_HISTORICAL_DATA"
            else:
                history = generate_synthetic_history(current_price, days=60, daily_vol=0.045, daily_drift=0.002)
                data_mode = "SYNTHETIC_DEMO_BENCHMARK"
        else:
            history = generate_synthetic_history(current_price, days=60, daily_vol=0.045, daily_drift=0.002)
            data_mode = "SYNTHETIC_DEMO_BENCHMARK"

        # 2. TVTP Markov state update
        state = markov.update(daily_ret=0.012, z_liq=1.1, z_trend=0.8, decoupling_active=decoupling)
        
        # 3. Rule 6 Capital Allocation (gated by synthetic check)
        is_synthetic = (data_mode == "SYNTHETIC_DEMO_BENCHMARK")
        alloc = self.reconciler.compute_allocation_weights(
            state, 
            decoupling_active=decoupling, 
            momentum_positive=True,
            is_synthetic_mode=is_synthetic
        )

        # 4. Foundation Model Statistical Prior
        tfm_prior = self.tfm_engine.forecast(history, horizon_days=horizon_days)

        # 5. Mechanism-Aware Scenario Corridor Conditioning
        cond = markov.condition_timesfm_quantiles(
            tfm_prior["p10_downside"], 
            tfm_prior["p50_expected"], 
            tfm_prior["p90_upside"], 
            state["state_vector"], 
            asset_vol_scale=0.045,
            horizon_steps=horizon_days,
            transition_matrix=state["transition_matrix"]
        )

        currency = "£" if "GBP" in ticker or "UK" in ticker else "$"
        falsify = f"Thesis falsified if price closes below {currency}{cond['downside_floor']:,.2f} on high stablecoin redemptions."
        
        summary = self.reconciler.generate_plain_english_summary(
            asset_name=f"{ticker} [{data_mode}]",
            current_price=current_price,
            currency_symbol=currency,
            forecast_output=cond,
            allocation_output=alloc,
            falsifiability_condition=falsify,
            raw_prior=tfm_prior
        )

        return {
            "asset_name": ticker,
            "current_price": current_price,
            "currency_symbol": currency,
            "data_mode": data_mode,
            "is_synthetic": is_synthetic,
            "history_length": len(history),
            "raw_prior": tfm_prior,
            "markov_state": state,
            "scenario_corridors": cond,
            "allocation": alloc,
            "falsifiability_condition": falsify,
            "summary": summary
        }

    def run_housing_pipeline(self, property_price=450000.0, postcode="NW1 4NP", history_file=None, history_series=None, horizon_days=365):
        markov = InstitutionalMarkovEngine(asset_daily_std=0.008)
        
        if history_series and len(history_series) > 0:
            history = list(history_series)
            property_price = history[-1]
            data_mode = "EMPIRICAL_HISTORICAL_DATA"
        elif history_file and os.path.exists(history_file):
            emp = load_history_series(history_file)
            if emp:
                history = emp
                property_price = history[-1]
                data_mode = "EMPIRICAL_HISTORICAL_DATA"
            else:
                history = generate_synthetic_history(property_price, days=180, daily_vol=0.008, daily_drift=0.0001)
                data_mode = "SYNTHETIC_DEMO_BENCHMARK"
        else:
            history = generate_synthetic_history(property_price, days=180, daily_vol=0.008, daily_drift=0.0001)
            data_mode = "SYNTHETIC_DEMO_BENCHMARK"

        state = markov.update(daily_ret=0.002, z_liq=0.1, z_trend=0.0, decoupling_active=False)
        is_synthetic = (data_mode == "SYNTHETIC_DEMO_BENCHMARK")
        alloc = self.reconciler.compute_allocation_weights(
            state, 
            decoupling_active=False, 
            momentum_positive=True,
            is_synthetic_mode=is_synthetic
        )

        tfm_prior = self.tfm_engine.forecast(history, horizon_days=horizon_days)
        cond = markov.condition_timesfm_quantiles(
            tfm_prior["p10_downside"], 
            tfm_prior["p50_expected"], 
            tfm_prior["p90_upside"], 
            state["state_vector"], 
            asset_vol_scale=0.008,
            horizon_steps=horizon_days,
            transition_matrix=state["transition_matrix"]
        )

        falsify = "Thesis falsified if local mortgage rates exceed 6.5% or regional transaction volume contracts > 30%."
        summary = self.reconciler.generate_plain_english_summary(
            asset_name=f"Residential Property ({postcode}) [{data_mode}]",
            current_price=property_price,
            currency_symbol="£",
            forecast_output=cond,
            allocation_output=alloc,
            falsifiability_condition=falsify,
            raw_prior=tfm_prior
        )

        return {
            "asset_name": f"Property ({postcode})",
            "current_price": property_price,
            "currency_symbol": "£",
            "postcode": postcode,
            "data_mode": data_mode,
            "is_synthetic": is_synthetic,
            "history_length": len(history),
            "raw_prior": tfm_prior,
            "markov_state": state,
            "scenario_corridors": cond,
            "allocation": alloc,
            "falsifiability_condition": falsify,
            "summary": summary
        }

    def run_portfolio_pipeline(self, holdings=None, holdings_str=None, file_path=None, history_file=None, horizon_days=30):
        target_holdings = {}
        if holdings and isinstance(holdings, dict):
            target_holdings = {k.upper(): float(v) for k, v in holdings.items()}
        elif file_path and os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                target_holdings = json.load(f)
        elif holdings_str:
            for item in holdings_str.split(","):
                if ":" in item:
                    k, v = item.split(":")
                    target_holdings[k.strip().upper()] = float(v.strip().replace("$", "").replace(",", ""))
        else:
            target_holdings = {"BTC": 4000.0, "ETH": 3000.0, "SOL": 1500.0, "USDT": 1500.0}

        total_value = sum(target_holdings.values())
        if total_value <= 0:
            raise ValueError("Total portfolio value must be greater than zero.")

        cash_tickers = {"USDT", "USDC", "DAI", "USD", "GBP", "EUR", "FDUSD", "USDE"}
        core_tickers = {"BTC", "ETH", "SOL"}
        
        cash_val = sum(v for k, v in target_holdings.items() if k in cash_tickers)
        core_val = sum(v for k, v in target_holdings.items() if k in core_tickers)
        spec_val = sum(v for k, v in target_holdings.items() if k not in cash_tickers and k not in core_tickers)

        cash_pct = (cash_val / total_value) * 100.0
        core_pct = (core_val / total_value) * 100.0
        spec_pct = (spec_val / total_value) * 100.0

        markov = InstitutionalMarkovEngine(asset_daily_std=0.045)
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        mcap_prov = self.liq_engine.get_stablecoin_mcap_with_provenance(today_str)
        mcap = mcap_prov["mcap"]
        decoupling = (mcap is not None and mcap > 150e9)

        state = markov.update(daily_ret=0.010, z_liq=0.9, z_trend=0.5, decoupling_active=decoupling)
        
        emp_history = load_history_series(history_file)
        if emp_history:
            history = emp_history
            total_value = history[-1]
            data_mode = "EMPIRICAL_HISTORICAL_DATA"
        else:
            history = generate_synthetic_history(total_value, days=60, daily_vol=0.035, daily_drift=0.0015)
            data_mode = "SYNTHETIC_DEMO_BENCHMARK"

        is_synthetic = (data_mode == "SYNTHETIC_DEMO_BENCHMARK")
        alloc = self.reconciler.compute_allocation_weights(
            state, 
            decoupling_active=decoupling, 
            momentum_positive=True,
            is_synthetic_mode=is_synthetic
        )

        tfm_prior = self.tfm_engine.forecast(history, horizon_days=horizon_days)
        cond = markov.condition_timesfm_quantiles(
            tfm_prior["p10_downside"], 
            tfm_prior["p50_expected"], 
            tfm_prior["p90_upside"], 
            state["state_vector"], 
            asset_vol_scale=0.035,
            horizon_steps=horizon_days,
            transition_matrix=state["transition_matrix"]
        )

        rebalance_notes = []
        if spec_pct > 35.0:
            excess_spec = spec_pct - 35.0
            rebalance_notes.append(f"Speculative altcoins ({spec_pct:.1f}%) exceed the 35% ceiling. Reallocate ${(excess_spec/100)*total_value:,.2f} to cash/real assets.")
        if cash_pct < 15.0:
            deficit_cash = 15.0 - cash_pct
            rebalance_notes.append(f"Cash buffer ({cash_pct:.1f}%) is below the 15% safety buffer. Target raising ${(deficit_cash/100)*total_value:,.2f} in dry powder.")

        directive = " | ".join(rebalance_notes) if rebalance_notes else "Portfolio allocation is compliant with Rule 6. Maintain trailing stops at Downside Scenario Floor."
        if is_synthetic:
            alloc["tactical_action"] = f"[ALLOCATION SUPPRESSED: SYNTHETIC CONE] {directive}"
        else:
            alloc["tactical_action"] = directive

        falsify = f"Thesis falsified if portfolio aggregate drops below ${cond['downside_floor']:,.2f} on macro stablecoin contraction."
        summary = self.reconciler.generate_plain_english_summary(
            asset_name=f"Custom Multi-Asset Portfolio ({len(target_holdings)} Assets) [{data_mode}]",
            current_price=total_value,
            currency_symbol="$",
            forecast_output=cond,
            allocation_output=alloc,
            falsifiability_condition=falsify,
            raw_prior=tfm_prior
        )

        return {
            "total_value": total_value,
            "holdings": target_holdings,
            "cash_pct": cash_pct,
            "core_pct": core_pct,
            "spec_pct": spec_pct,
            "excess_spec_pct": max(0.0, spec_pct - 35.0),
            "deficit_cash_pct": max(0.0, 15.0 - cash_pct),
            "data_mode": data_mode,
            "is_synthetic": is_synthetic,
            "raw_prior": tfm_prior,
            "markov_state": state,
            "scenario_corridors": cond,
            "allocation": alloc,
            "falsifiability_condition": falsify,
            "summary": summary
        }

    def run_media_pipeline(self, monthly_spend=10000.0, cpm=12.50, ec50_spend=15000.0, k_max_impressions=2500000.0, gamma=1.3):
        # Hill Saturation Model:
        # Response(S) = K_max * (S^gamma / (EC50^gamma + S^gamma))
        denom = (ec50_spend ** gamma) + (monthly_spend ** gamma)
        saturated_impressions = k_max_impressions * ((monthly_spend ** gamma) / denom)
        
        # Marginal yield derivative: dR/dS
        marginal_yield = k_max_impressions * (gamma * (monthly_spend ** (gamma - 1)) * (ec50_spend ** gamma)) / (denom ** 2)
        effective_cpm = (monthly_spend / max(1.0, saturated_impressions)) * 1000.0
        marginal_cpm = (1000.0 / marginal_yield) if marginal_yield > 0 else 999.0

        markov = InstitutionalMarkovEngine(asset_daily_std=0.025)
        state = markov.update(daily_ret=0.005, z_liq=0.5, z_trend=0.2, decoupling_active=False)

        floor_yield = saturated_impressions * 0.88  # Ad fatigue / tracking loss
        expected_yield = saturated_impressions * 1.02  # Expected organic + paid yield
        ceiling_yield = saturated_impressions * 1.18  # Algorithmic distribution lift

        pacing_status = "OPTIMAL_PACING" if marginal_cpm < (cpm * 1.5) else "SATURATION_WARNING"
        pacing_action = (
            f"OPTIMAL PACING: Marginal CPM is ${marginal_cpm:.2f} (Effective Blended CPM: ${effective_cpm:.2f}). Spend is within efficient return zone."
            if pacing_status == "OPTIMAL_PACING" else
            f"SATURATION DIRECTIVE: Diminishing returns severe! Marginal CPM has spiked to ${marginal_cpm:.2f}. Cap budget near ${ec50_spend:,.0f} to avoid ad fatigue."
        )

        alloc = {
            "target_risk_weight": 0.70 if pacing_status == "OPTIMAL_PACING" else 0.40,
            "cash_buffer_weight": 0.30 if pacing_status == "OPTIMAL_PACING" else 0.60,
            "regime": pacing_status,
            "tactical_action": pacing_action,
            "fragility_score": 5.0,
            "ponzi_probability": 0.05,
            "allocation_disabled": False
        }

        cond = {
            "downside_floor": round(floor_yield, 2),
            "expected_target": round(expected_yield, 2),
            "upside_ceiling": round(ceiling_yield, 2),
            "reconciled_p10": round(floor_yield, 2),
            "reconciled_p50": round(expected_yield, 2),
            "reconciled_p90": round(ceiling_yield, 2)
        }

        falsify = f"Media model falsified if Blended CPM exceeds ${(cpm * 1.35):,.2f} or CTR drops below 0.85%."
        raw_summary = self.reconciler.generate_plain_english_summary(
            asset_name=f"Media Campaign (${monthly_spend:,.0f}/mo budget; EC50=${ec50_spend:,.0f}, K_max={k_max_impressions:,.0f})",
            current_price=saturated_impressions,
            currency_symbol="",
            forecast_output=cond,
            allocation_output=alloc,
            falsifiability_condition=falsify
        )
        clean_summary = raw_summary.replace("$", "").replace("Current Price: ", "Estimated Monthly Yield: ") + " impressions"

        return {
            "monthly_spend": monthly_spend,
            "cpm": cpm,
            "ec50_spend": ec50_spend,
            "k_max_impressions": k_max_impressions,
            "saturated_impressions": saturated_impressions,
            "effective_cpm": effective_cpm,
            "marginal_cpm": marginal_cpm,
            "pacing_status": pacing_status,
            "pacing_action": pacing_action,
            "scenario_corridors": cond,
            "allocation": alloc,
            "falsifiability_condition": falsify,
            "summary": clean_summary
        }
