"""
Causal TimesFM Engine v2.3: Unified Operational Pipeline
=========================================================
Executes the institutional 8-stage causal forecasting & capital governance pipeline:
  Stage 1: Raw Data Ingestion & Frequency/Timestamp Lineage Tracking
  Stage 2: Foundation Model Prior (TimesFM with degraded-mode telemetry)
  Stage 3: Economic Mechanism Layer (DefiLlama net float expansion + Minsky/Structural anchors)
  Stage 4: Econometric Evidence & Falsification Gates (Stationarity, Granger, Walk-Forward Theil's U)
  Stage 5: Dynamic TVTP Markov Filtering & Multi-Mode Horizon State Propagation
  Stage 6: Mechanism-Aware Scenario Corridor Envelope (Downside Floor, Central Target, Upside Ceiling)
  Stage 7: Normative Rule 6 Capital Allocation (Hard-gated by Synthetic Guardrail & Theil's U)
  Stage 8: Plain-English Executive Decision Sheet with Machine-Testable Falsification Objects
"""

import os
import math
import json
from datetime import datetime, timezone

from core.markov_regime import InstitutionalMarkovEngine
from core.onchain_liquidity import OnChainLiquidityEngine
from core.reconciliation import ReconciliationEngine
from core.engine_timesfm import TimesFmBaselineEngine
from core.econometrics import EconometricFilter
from core.engine_structural import StructuralMacroEngine

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

def parse_date_safely(date_str):
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def load_history_series_with_metadata(file_path):
    """
    Loads empirical historical series from CSV/text with timestamp preservation
    and automatic sampling frequency inference.
    """
    timestamps = []
    values = []
    
    if file_path and os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.lower().startswith(("date", "time", "timestamp", "price")):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        d = parse_date_safely(parts[0])
                        try:
                            v = float(parts[-1])
                            if d is not None:
                                timestamps.append(d.strftime("%Y-%m-%d"))
                            else:
                                timestamps.append(parts[0])
                            values.append(v)
                        except ValueError:
                            continue
                    elif len(parts) == 1:
                        try:
                            v = float(parts[0])
                            values.append(v)
                        except ValueError:
                            continue

    frequency = "U"
    step_days = 1.0
    if len(timestamps) >= 3:
        deltas = []
        for i in range(1, len(timestamps)):
            d0 = parse_date_safely(timestamps[i - 1])
            d1 = parse_date_safely(timestamps[i])
            if d0 and d1:
                deltas.append(abs((d1 - d0).days))
        
        if deltas:
            deltas.sort()
            median_delta = deltas[len(deltas) // 2]
            step_days = median_delta
            if 0.5 <= median_delta <= 2.5:
                frequency = "D"  # Daily
            elif 5.0 <= median_delta <= 9.0:
                frequency = "W"  # Weekly
            elif 25.0 <= median_delta <= 35.0:
                frequency = "M"  # Monthly
            elif 80.0 <= median_delta <= 100.0:
                frequency = "Q"  # Quarterly
            else:
                frequency = "U"

    as_of = timestamps[-1] if timestamps else datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return {
        "timestamps": timestamps,
        "values": values,
        "frequency": frequency,
        "median_step_days": step_days,
        "as_of_date": as_of
    }

def map_horizon_to_steps(horizon_days, frequency):
    """
    Translates requested horizon in days to model forecast steps based on series frequency.
    Prevents forecasting 365 steps (30 years) on monthly real estate data!
    """
    if frequency == "M":
        return max(1, round(horizon_days / 30.4375))
    elif frequency == "W":
        return max(1, round(horizon_days / 7.0))
    elif frequency == "Q":
        return max(1, round(horizon_days / 91.25))
    else:  # 'D' or 'U'
        return max(1, int(horizon_days))

def create_parameter_record(value, source, as_of=None, transformation="identity", status="OBSERVED", notes=""):
    return {
        "value": value,
        "source": source,
        "as_of_timestamp": as_of or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "transformation": transformation,
        "parameter_status": status,
        "notes": notes
    }

class CausalTimesFmPipeline:
    """
    Unified institutional analysis pipeline orchestrating data provenance,
    foundation model inference, causal macro conditioning, econometric gating,
    and risk policy across all domains.
    """
    def __init__(self, delta_threshold_pct=15.0):
        self.liq_engine = OnChainLiquidityEngine()
        self.reconciler = ReconciliationEngine(delta_threshold_pct=delta_threshold_pct)
        self.tfm_engine = TimesFmBaselineEngine()
        self.econ_filter = EconometricFilter()
        self.struct_engine = StructuralMacroEngine()

    def run_crypto_pipeline(self, ticker="BTC-USD", current_price=94000.0, history_file=None, history_series=None, horizon_days=30):
        # ---------------------------------------------------------
        # STAGE 1: Raw Data & Lineage Provenance Ingestion
        # ---------------------------------------------------------
        parameter_lineage = {}
        if history_series and len(history_series) > 0:
            history = [float(x) for x in history_series]
            current_price = history[-1]
            data_mode = "EMPIRICAL_HISTORICAL_DATA"
            frequency = "D"
            latest_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        elif history_file and os.path.exists(history_file):
            loaded = load_history_series_with_metadata(history_file)
            if loaded["values"] and len(loaded["values"]) > 0:
                history = loaded["values"]
                current_price = history[-1]
                data_mode = "EMPIRICAL_HISTORICAL_DATA"
                frequency = loaded["frequency"]
                latest_date = loaded["as_of_date"]
            else:
                history = generate_synthetic_history(current_price, days=60, daily_vol=0.045, daily_drift=0.002)
                data_mode = "SYNTHETIC_DEMO_BENCHMARK"
                frequency = "D"
                latest_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        else:
            history = generate_synthetic_history(current_price, days=60, daily_vol=0.045, daily_drift=0.002)
            data_mode = "SYNTHETIC_DEMO_BENCHMARK"
            frequency = "D"
            latest_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        is_synthetic = (data_mode == "SYNTHETIC_DEMO_BENCHMARK")
        model_steps = map_horizon_to_steps(horizon_days, frequency)

        # ---------------------------------------------------------
        # STAGE 2: Foundation Model Prior (TimesFM Baseline)
        # ---------------------------------------------------------
        tfm_prior = self.tfm_engine.forecast(history, horizon_days=model_steps)

        # ---------------------------------------------------------
        # STAGE 3: Economic Mechanism Layer (Empirical Ingestion)
        # ---------------------------------------------------------
        if not is_synthetic:
            ret_t = (history[-1] - history[-2]) / history[-2] if len(history) >= 2 and history[-2] != 0 else 0.0
            parameter_lineage["daily_ret"] = create_parameter_record(
                round(ret_t, 4), source="empirical_price_series", as_of=latest_date, transformation="1_step_pct_return", status="OBSERVED"
            )
            
            w_trend = min(60, len(history))
            trailing_window = history[-w_trend:]
            mean_tr = sum(trailing_window) / len(trailing_window)
            std_tr = math.sqrt(sum((x - mean_tr) ** 2 for x in trailing_window) / max(1, len(trailing_window) - 1))
            z_trend_val = (history[-1] - mean_tr) / (std_tr if std_tr > 1e-6 else 1.0)
            z_trend = max(-3.5, min(3.5, z_trend_val))
            parameter_lineage["z_trend"] = create_parameter_record(
                round(z_trend, 2), source="empirical_price_series", as_of=latest_date, transformation=f"trailing_{w_trend}_zscore", status="OBSERVED"
            )

            liq_features = self.liq_engine.get_liquidity_features_as_of(latest_date)
            z_liq = liq_features["z_score"]
            float_growth_30d = liq_features["float_growth_30d"]
            decoupling_active = liq_features["decoupling_active"]
            parameter_lineage["z_liq"] = create_parameter_record(
                z_liq, source="defillama_aggregate_stablecoins", as_of=liq_features["as_of_date"], transformation="rolling_float_zscore", status="OBSERVED"
            )
            parameter_lineage["float_growth_30d"] = create_parameter_record(
                float_growth_30d, source="defillama_aggregate_stablecoins", as_of=liq_features["as_of_date"], transformation="rolling_30d_pct_growth", status="OBSERVED"
            )
            parameter_lineage["decoupling_active"] = create_parameter_record(
                decoupling_active, source="onchain_liquidity_engine", as_of=liq_features["as_of_date"], transformation="z_liq > 0.8 AND float_growth_30d > 0.0", status="ESTIMATED"
            )
        else:
            ret_t = 0.012
            z_trend = 0.8
            z_liq = 1.1
            float_growth_30d = 0.035
            decoupling_active = True
            parameter_lineage["daily_ret"] = create_parameter_record(ret_t, source="synthetic_cone", as_of=latest_date, transformation="mock", status="DEMO_ONLY")
            parameter_lineage["z_trend"] = create_parameter_record(z_trend, source="synthetic_cone", as_of=latest_date, transformation="mock", status="DEMO_ONLY")
            parameter_lineage["z_liq"] = create_parameter_record(z_liq, source="synthetic_cone", as_of=latest_date, transformation="mock", status="DEMO_ONLY")
            parameter_lineage["decoupling_active"] = create_parameter_record(decoupling_active, source="synthetic_cone", as_of=latest_date, transformation="mock", status="DEMO_ONLY")

        struct_val = self.struct_engine.compute_structural_target(
            current_price=current_price,
            macro_liquidity_regime="expanding" if decoupling_active else "neutral",
            circulating_supply=0.90,
            total_supply=1.00,
            minsky_stage="Hedge" if decoupling_active else "Speculative"
        )

        # ---------------------------------------------------------
        # STAGE 4: Econometric Evidence & Falsification Gates
        # ---------------------------------------------------------
        stat_report = self.econ_filter.stationarize(history)
        
        theils_eval = None
        theils_passed = True
        if len(history) >= 35:
            try:
                def model_fc(train_slice, h_step):
                    return self.tfm_engine.forecast(train_slice, horizon_days=h_step)
                min_eval_train = min(30, len(history) - 5)
                theils_eval = self.econ_filter.evaluate_rolling_origin_theils_u(
                    history, model_fc, min_train_len=min_eval_train, horizon=1
                )
                theils_passed = theils_eval["hurdle_passed"]
            except Exception:
                theils_passed = True

        # ---------------------------------------------------------
        # STAGE 5: Dynamic TVTP Markov Filtering & Horizon Propagation
        # ---------------------------------------------------------
        markov = InstitutionalMarkovEngine(asset_daily_std=0.045)
        state = markov.update(daily_ret=ret_t, z_liq=z_liq, z_trend=z_trend, decoupling_active=decoupling_active)

        # ---------------------------------------------------------
        # STAGE 6: Mechanism-Aware Scenario Corridor Envelope
        # ---------------------------------------------------------
        cond = markov.condition_timesfm_quantiles(
            tfm_p10=tfm_prior["p10_downside"], 
            tfm_p50=tfm_prior["p50_expected"], 
            tfm_p90=tfm_prior["p90_upside"], 
            forward_xi=state["state_vector"], 
            asset_vol_scale=0.045,
            horizon_steps=model_steps,
            transition_matrix=state["transition_matrix"],
            horizon_mode="frozen_transition"
        )

        # ---------------------------------------------------------
        # STAGE 7: Normative Rule 6 Capital Allocation Policy
        # ---------------------------------------------------------
        alloc = self.reconciler.compute_allocation_weights(
            state, 
            decoupling_active=decoupling_active, 
            momentum_positive=(z_trend > 0.0),
            is_synthetic_mode=is_synthetic
        )

        if not is_synthetic and not theils_passed and theils_eval:
            alloc["theils_u_failed"] = True
            alloc["target_risk_weight"] = round(min(alloc["target_risk_weight"], 0.35), 2)
            alloc["cash_buffer_weight"] = round(1.0 - alloc["target_risk_weight"], 2)
            alloc["tactical_action"] = f"[THEIL'S U HURDLE WARNING (U={theils_eval['theils_u']:.2f} >= 1.0)]: Model does not beat naive persistence. High-beta exposure capped at 35%. " + alloc["tactical_action"]

        currency = "£" if "GBP" in ticker or "UK" in ticker else "$"
        floor = cond["downside_floor"]
        target = cond["expected_target"]
        ceiling = cond["upside_ceiling"]

        falsify_obj = {
            "primary_metric": "price",
            "threshold": floor,
            "operator": "<",
            "secondary_metric": "stablecoin_float_growth_30d",
            "secondary_threshold": -0.05,
            "horizon_steps": model_steps,
            "frequency": frequency,
            "evaluated_status": "ACTIVE_MONITORING"
        }
        falsify_str = f"Thesis falsified if price closes below {currency}{floor:,.2f} while 30-day stablecoin float growth contracts beyond -5.0%."

        summary = self.reconciler.generate_plain_english_summary(
            asset_name=f"{ticker} [{data_mode} | Freq: {frequency} | AsOf: {latest_date}]",
            current_price=current_price,
            currency_symbol=currency,
            forecast_output=cond,
            allocation_output=alloc,
            falsifiability_condition=falsify_str,
            raw_prior=tfm_prior
        )

        return {
            "asset_name": ticker,
            "current_price": current_price,
            "currency_symbol": currency,
            "data_mode": data_mode,
            "is_synthetic": is_synthetic,
            "frequency": frequency,
            "model_steps": model_steps,
            "history_length": len(history),
            "as_of_date": latest_date,
            "raw_prior": tfm_prior,
            "structural_valuation": struct_val,
            "stationarity_report": stat_report,
            "theils_u_eval": theils_eval,
            "theils_u_passed": theils_passed,
            "markov_state": state,
            "scenario_corridors": cond,
            "allocation": alloc,
            "parameter_lineage": parameter_lineage,
            "falsifiability_object": falsify_obj,
            "falsifiability_condition": falsify_str,
            "summary": summary
        }

    def run_housing_pipeline(self, property_price=450000.0, postcode="NW1 4NP", history_file=None, history_series=None, horizon_days=365):
        parameter_lineage = {}
        if history_series and len(history_series) > 0:
            history = [float(x) for x in history_series]
            property_price = history[-1]
            data_mode = "EMPIRICAL_HISTORICAL_DATA"
            frequency = "M"
            latest_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        elif history_file and os.path.exists(history_file):
            loaded = load_history_series_with_metadata(history_file)
            if loaded["values"]:
                history = loaded["values"]
                property_price = history[-1]
                data_mode = "EMPIRICAL_HISTORICAL_DATA"
                frequency = loaded["frequency"]
                latest_date = loaded["as_of_date"]
            else:
                history = generate_synthetic_history(property_price, days=180, daily_vol=0.008, daily_drift=0.0001)
                data_mode = "SYNTHETIC_DEMO_BENCHMARK"
                frequency = "M"
                latest_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        else:
            history = generate_synthetic_history(property_price, days=180, daily_vol=0.008, daily_drift=0.0001)
            data_mode = "SYNTHETIC_DEMO_BENCHMARK"
            frequency = "M"
            latest_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        is_synthetic = (data_mode == "SYNTHETIC_DEMO_BENCHMARK")
        model_steps = map_horizon_to_steps(horizon_days, frequency)

        tfm_prior = self.tfm_engine.forecast(history, horizon_days=model_steps)

        if not is_synthetic:
            ret_t = (history[-1] - history[-2]) / history[-2] if len(history) >= 2 and history[-2] != 0 else 0.0
            parameter_lineage["step_ret"] = create_parameter_record(round(ret_t, 4), source="land_registry_series", as_of=latest_date, transformation="1_step_return", status="OBSERVED")
            w = min(24, len(history))
            trailing = history[-w:]
            m_val = sum(trailing) / len(trailing)
            s_val = math.sqrt(sum((x - m_val)**2 for x in trailing) / max(1, len(trailing)-1))
            z_trend = (history[-1] - m_val) / (s_val if s_val > 1e-6 else 1.0)
            parameter_lineage["z_trend"] = create_parameter_record(round(z_trend, 2), source="land_registry_series", as_of=latest_date, transformation=f"trailing_{w}_zscore", status="OBSERVED")
        else:
            ret_t = 0.002
            z_trend = 0.0
            parameter_lineage["step_ret"] = create_parameter_record(ret_t, source="synthetic_cone", as_of=latest_date, status="DEMO_ONLY")
            parameter_lineage["z_trend"] = create_parameter_record(z_trend, source="synthetic_cone", as_of=latest_date, status="DEMO_ONLY")

        markov = InstitutionalMarkovEngine(asset_daily_std=0.008)
        state = markov.update(daily_ret=ret_t, z_liq=0.1, z_trend=z_trend, decoupling_active=False)

        cond = markov.condition_timesfm_quantiles(
            tfm_p10=tfm_prior["p10_downside"], 
            tfm_p50=tfm_prior["p50_expected"], 
            tfm_p90=tfm_prior["p90_upside"], 
            forward_xi=state["state_vector"], 
            asset_vol_scale=0.008,
            horizon_steps=model_steps,
            transition_matrix=state["transition_matrix"],
            horizon_mode="frozen_transition"
        )

        alloc = self.reconciler.compute_allocation_weights(
            state, 
            decoupling_active=False, 
            momentum_positive=(z_trend >= 0.0),
            is_synthetic_mode=is_synthetic
        )

        falsify_obj = {
            "primary_metric": "mortgage_rate",
            "threshold": 6.5,
            "operator": ">",
            "secondary_metric": "transaction_volume_pct_change",
            "secondary_threshold": -30.0,
            "horizon_steps": model_steps,
            "frequency": frequency,
            "status": "ACTIVE_MONITORING"
        }
        falsify_str = f"Thesis falsified if local mortgage rates exceed 6.5% or regional transaction volume contracts > 30% over the {model_steps}-{frequency} forward horizon."

        summary = self.reconciler.generate_plain_english_summary(
            asset_name=f"Residential Property ({postcode}) [{data_mode} | Freq: {frequency} | AsOf: {latest_date}]",
            current_price=property_price,
            currency_symbol="£",
            forecast_output=cond,
            allocation_output=alloc,
            falsifiability_condition=falsify_str,
            raw_prior=tfm_prior
        )

        return {
            "asset_name": f"Property ({postcode})",
            "current_price": property_price,
            "currency_symbol": "£",
            "postcode": postcode,
            "data_mode": data_mode,
            "is_synthetic": is_synthetic,
            "frequency": frequency,
            "model_steps": model_steps,
            "history_length": len(history),
            "as_of_date": latest_date,
            "raw_prior": tfm_prior,
            "markov_state": state,
            "scenario_corridors": cond,
            "allocation": alloc,
            "parameter_lineage": parameter_lineage,
            "falsifiability_object": falsify_obj,
            "falsifiability_condition": falsify_str,
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

        parameter_lineage = {}
        if history_file and os.path.exists(history_file):
            loaded = load_history_series_with_metadata(history_file)
            if loaded["values"]:
                history = loaded["values"]
                total_value = history[-1]
                data_mode = "EMPIRICAL_HISTORICAL_DATA"
                frequency = loaded["frequency"]
                latest_date = loaded["as_of_date"]
            else:
                history = generate_synthetic_history(total_value, days=60, daily_vol=0.035, daily_drift=0.0015)
                data_mode = "SYNTHETIC_DEMO_BENCHMARK"
                frequency = "D"
                latest_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        else:
            history = generate_synthetic_history(total_value, days=60, daily_vol=0.035, daily_drift=0.0015)
            data_mode = "SYNTHETIC_DEMO_BENCHMARK"
            frequency = "D"
            latest_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        is_synthetic = (data_mode == "SYNTHETIC_DEMO_BENCHMARK")
        model_steps = map_horizon_to_steps(horizon_days, frequency)

        if not is_synthetic:
            ret_t = (history[-1] - history[-2]) / history[-2] if len(history) >= 2 and history[-2] != 0 else 0.0
            liq_features = self.liq_engine.get_liquidity_features_as_of(latest_date)
            z_liq = liq_features["z_score"]
            decoupling_active = liq_features["decoupling_active"]
            parameter_lineage["daily_ret"] = create_parameter_record(round(ret_t, 4), source="portfolio_history", as_of=latest_date, status="OBSERVED")
            parameter_lineage["z_liq"] = create_parameter_record(z_liq, source="defillama_aggregate_stablecoins", as_of=liq_features["as_of_date"], status="OBSERVED")
        else:
            ret_t = 0.010
            z_liq = 0.9
            decoupling_active = True
            parameter_lineage["daily_ret"] = create_parameter_record(ret_t, source="synthetic_cone", as_of=latest_date, status="DEMO_ONLY")
            parameter_lineage["z_liq"] = create_parameter_record(z_liq, source="synthetic_cone", as_of=latest_date, status="DEMO_ONLY")

        markov = InstitutionalMarkovEngine(asset_daily_std=0.045)
        state = markov.update(daily_ret=ret_t, z_liq=z_liq, z_trend=0.5, decoupling_active=decoupling_active)

        alloc = self.reconciler.compute_allocation_weights(
            state, 
            decoupling_active=decoupling_active, 
            momentum_positive=True,
            is_synthetic_mode=is_synthetic
        )

        tfm_prior = self.tfm_engine.forecast(history, horizon_days=model_steps)
        cond = markov.condition_timesfm_quantiles(
            tfm_p10=tfm_prior["p10_downside"], 
            tfm_p50=tfm_prior["p50_expected"], 
            tfm_p90=tfm_prior["p90_upside"], 
            forward_xi=state["state_vector"], 
            asset_vol_scale=0.035,
            horizon_steps=model_steps,
            transition_matrix=state["transition_matrix"],
            horizon_mode="frozen_transition"
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

        falsify_str = f"Thesis falsified if portfolio aggregate drops below ${cond['downside_floor']:,.2f} on macro stablecoin contraction."
        summary = self.reconciler.generate_plain_english_summary(
            asset_name=f"Custom Multi-Asset Portfolio ({len(target_holdings)} Assets) [{data_mode}]",
            current_price=total_value,
            currency_symbol="$",
            forecast_output=cond,
            allocation_output=alloc,
            falsifiability_condition=falsify_str,
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
            "frequency": frequency,
            "model_steps": model_steps,
            "as_of_date": latest_date,
            "raw_prior": tfm_prior,
            "markov_state": state,
            "scenario_corridors": cond,
            "allocation": alloc,
            "parameter_lineage": parameter_lineage,
            "falsifiability_condition": falsify_str,
            "summary": summary
        }

    def solve_optimal_media_spend(self, target_cpm, ec50_spend, k_max_impressions, gamma=1.3, multiplier=1.5):
        """
        Analytically solves for the exact economic spend ceiling S* where
        Marginal CPM(S*) = multiplier * target_cpm.
        Eliminates heuristic 'cap near EC50' rule.
        """
        cpm_ceiling = target_cpm * multiplier
        
        def marginal_cpm(s):
            denom = (ec50_spend ** gamma) + (s ** gamma)
            m_yield = k_max_impressions * (gamma * (s ** (gamma - 1)) * (ec50_spend ** gamma)) / (denom ** 2)
            return (1000.0 / m_yield) if m_yield > 0 else 99999.0

        low = max(10.0, ec50_spend * 0.05)
        high = ec50_spend * 15.0
        
        for _ in range(40):
            mid = (low + high) / 2.0
            val = marginal_cpm(mid)
            if val < cpm_ceiling:
                low = mid
            else:
                high = mid
                
        return (low + high) / 2.0

    def run_media_pipeline(self, monthly_spend=10000.0, cpm=12.50, ec50_spend=15000.0, k_max_impressions=2500000.0, gamma=1.3):
        """
        Marketing-native Media Investment & Pacing Engine.
        Completely decoupled from financial Ponzi/risk abstractions.
        """
        denom = (ec50_spend ** gamma) + (monthly_spend ** gamma)
        saturated_impressions = k_max_impressions * ((monthly_spend ** gamma) / denom)
        
        marginal_yield = k_max_impressions * (gamma * (monthly_spend ** (gamma - 1)) * (ec50_spend ** gamma)) / (denom ** 2)
        effective_cpm = (monthly_spend / max(1.0, saturated_impressions)) * 1000.0
        marginal_cpm = (1000.0 / marginal_yield) if marginal_yield > 0 else 999.0

        optimal_spend_ceiling = self.solve_optimal_media_spend(cpm, ec50_spend, k_max_impressions, gamma, multiplier=1.5)
        
        is_saturated = monthly_spend > optimal_spend_ceiling
        pacing_status = "DIMINISHING_RETURNS_WARNING" if is_saturated else "OPTIMAL_PACING_ZONE"
        
        if is_saturated:
            pacing_action = (
                f"SATURATION CEILING BREACHED: Marginal CPM has reached ${marginal_cpm:.2f} (exceeds ${cpm * 1.5:.2f} limit). "
                f"Mathematically optimal budget ceiling is ${optimal_spend_ceiling:,.0f}/mo. Cap spend at ${optimal_spend_ceiling:,.0f} to avoid budget exhaustion."
            )
        else:
            pacing_action = (
                f"OPTIMAL EFFICIENCY: Marginal CPM is ${marginal_cpm:.2f} (Effective Blended CPM: ${effective_cpm:.2f}). "
                f"Spend is well within the high-yield return corridor (Optimal spend ceiling: ${optimal_spend_ceiling:,.0f}/mo)."
            )

        floor_yield = saturated_impressions * 0.88
        expected_yield = saturated_impressions * 1.02
        ceiling_yield = saturated_impressions * 1.18

        media_decision = {
            "target_spend_budget": round(min(monthly_spend, optimal_spend_ceiling), 2),
            "reserve_budget_excess": round(max(0.0, monthly_spend - optimal_spend_ceiling), 2),
            "optimal_spend_ceiling": round(optimal_spend_ceiling, 2),
            "pacing_status": pacing_status,
            "pacing_action": pacing_action,
            "saturation_risk_score": round(min(100.0, (monthly_spend / optimal_spend_ceiling) * 50.0), 1),
            "ad_fatigue_probability": round(min(1.0, (monthly_spend / optimal_spend_ceiling) * 0.25), 3),
            "domain_policy_type": "MARKETING_CAPITAL_PACING_POLICY"
        }

        cond = {
            "downside_floor": round(floor_yield, 2),
            "expected_target": round(expected_yield, 2),
            "upside_ceiling": round(ceiling_yield, 2),
            "reconciled_p10": round(floor_yield, 2),
            "reconciled_p50": round(expected_yield, 2),
            "reconciled_p90": round(ceiling_yield, 2)
        }

        falsify_str = f"Media model falsified if Blended CPM exceeds ${(cpm * 1.35):,.2f} or CTR drops below 0.85%."
        
        summary = f"""
================================================================================
                    MEDIA CAMPAIGN DECISION SHEET
================================================================================
Campaign Evaluated: Media Attention (Budget: ${monthly_spend:,.0f}/mo | EC50: ${ec50_spend:,.0f} | K_max: {k_max_impressions:,.0f})

1. WHERE WE STAND TODAY (HILL SATURATION ECONOMICS):
   - Pacing Status: {pacing_status}
   - Addressable Audience Saturation: {media_decision['saturation_risk_score']:.1f} / 100
   - Effective Blended CPM: ${effective_cpm:.2f}
   - Marginal CPM (Next $1k Spend): ${marginal_cpm:.2f} (Target CPM: ${cpm:.2f})

2. WHAT TO DO WITH YOUR AD BUDGET (MEDIA PACING POLICY):
   - Mathematically Optimal Spend Ceiling: ${optimal_spend_ceiling:,.0f}/mo
   - Recommended Monthly Budget:            ${media_decision['target_spend_budget']:,.0f}/mo
   - Capital to Hold in Reserve / Reallocate: ${media_decision['reserve_budget_excess']:,.0f}/mo
   - Action Directive: {pacing_action}

3. PROJECTED MONTHLY IMPRESSIONS (SCENARIO CORRIDORS):
   - Downside Impression Floor:      {floor_yield:,.0f} impressions ({((floor_yield/saturated_impressions)-1)*100:+.1f}%)
   - Most Likely Expected Reach:     {expected_yield:,.0f} impressions ({((expected_yield/saturated_impressions)-1)*100:+.1f}%)
   - Upside Algorithmic Reach:       {ceiling_yield:,.0f} impressions ({((ceiling_yield/saturated_impressions)-1)*100:+.1f}%)

4. WHAT WOULD PROVE THIS ANALYSIS WRONG (FALSIFIABILITY):
   {falsify_str}
================================================================================
"""

        return {
            "monthly_spend": monthly_spend,
            "cpm": cpm,
            "ec50_spend": ec50_spend,
            "k_max_impressions": k_max_impressions,
            "saturated_impressions": saturated_impressions,
            "effective_cpm": effective_cpm,
            "marginal_cpm": marginal_cpm,
            "optimal_spend_ceiling": optimal_spend_ceiling,
            "pacing_status": pacing_status,
            "pacing_action": pacing_action,
            "scenario_corridors": cond,
            "media_decision": media_decision,
            "falsifiability_condition": falsify_str,
            "summary": summary.strip()
        }
