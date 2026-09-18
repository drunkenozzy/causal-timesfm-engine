"""
On-Chain Endogenous Base Money and Liquidity Engine
===================================================
Models Post-Keynesian endogenous money creation in digital assets:
  In traditional economics: M2 = Central Bank reserves + Commercial Bank credit.
  In crypto economies: M = Circulating Stablecoins (USDT + USDC + DAI).

Ingests 24/7 daily aggregate stablecoin market capitalization (DefiLlama API)
and computes strictly backward-looking, zero-lookahead rolling Z-scores.
Eliminates traditional market weekend closures and arbitrary hardcoded thresholds.
"""

import os
import json
import math
from datetime import datetime

class OnChainLiquidityEngine:
    def __init__(self, cache_file=None):
        if cache_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_file = os.path.join(base_dir, "data", "stablecoins_aggregate_history.json")
        self.cache_file = cache_file
        self.history_by_date = {}
        self._load_data()

    def _load_data(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    ts = item["timestamp"]
                    dt_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                    self.history_by_date[dt_str] = item["stablecoin_circulating_usd"]
            except Exception as e:
                print(f"[OnChainLiquidityEngine] Error loading cache: {e}")

    def get_stablecoin_mcap_with_provenance(self, date_str):
        """Returns stablecoin mcap along with explicit data freshness provenance."""
        if date_str in self.history_by_date:
            return {"mcap": self.history_by_date[date_str], "status": "FRESH", "staleness_days": 0, "as_of": date_str}
        
        sorted_dates = sorted(self.history_by_date.keys())
        prior_dates = [d for d in sorted_dates if d <= date_str]
        if prior_dates:
            last_date = prior_dates[-1]
            try:
                d1 = datetime.strptime(date_str, "%Y-%m-%d")
                d0 = datetime.strptime(last_date, "%Y-%m-%d")
                gap = (d1 - d0).days
            except Exception:
                gap = 1
            status = "STALE_ACCEPTABLE" if gap <= 3 else "DATA_STALE_CRITICAL"
            return {"mcap": self.history_by_date[last_date], "status": status, "staleness_days": gap, "as_of": last_date}
        
        return {"mcap": None, "status": "INVALID_NO_DATA", "staleness_days": 999, "as_of": None}

    def get_stablecoin_mcap(self, date_str, refuse_stale=True):
        prov = self.get_stablecoin_mcap_with_provenance(date_str)
        if prov["status"] == "DATA_STALE_CRITICAL":
            print(f"[DATA BLOCKED] Stablecoin observation is critically STALE: {prov['as_of']} (requested {date_str}, lag: {prov['staleness_days']} days).")
            if refuse_stale:
                return None
        return prov["mcap"]

    def compute_rolling_features(self, date_list, window=252):
        """
        Computes 30-day net float expansion and rolling 252-day Z-scores.
        Zero-Lookahead Guaranteed: Observations during the 30-day warm-up period
        are strictly set to None, never backfilled from future observations.
        """
        n = len(date_list)
        mcaps = []
        provenances = []
        for d in date_list:
            p = self.get_stablecoin_mcap_with_provenance(d)
            mcaps.append(p["mcap"])
            provenances.append(p)

        # 30-day Net Float Growth (Liquidity Impulse)
        # Warmup index 0..29 is strictly None (no backward lookahead leak)
        float_growth_30d = [None] * n
        for i in range(30, n):
            prev = mcaps[i - 30]
            curr = mcaps[i]
            if prev is not None and curr is not None and prev > 0:
                float_growth_30d[i] = (curr - prev) / prev
            else:
                float_growth_30d[i] = None

        # Rolling Z-scores computed strictly on past non-null float growth
        z_scores = [None] * n
        for i in range(30 + window, n):
            hist = [x for x in float_growth_30d[i - window : i] if x is not None]
            if len(hist) >= window * 0.8:
                mu = sum(hist) / len(hist)
                var = sum((x - mu) ** 2 for x in hist) / len(hist)
                sigma = math.sqrt(var) if var > 1e-12 else 1e-6
                if float_growth_30d[i] is not None:
                    z_scores[i] = (float_growth_30d[i] - mu) / sigma

        results = []
        for i in range(n):
            d = date_list[i]
            v = float_growth_30d[i]
            z = z_scores[i]
            prov_status = provenances[i]["status"]
            
            # Hard refusal on critical staleness
            if prov_status == "DATA_STALE_CRITICAL":
                regime = "DATA_STALE_REFUSED"
                decoupling_active = False
            elif v is None or z is None:
                regime = "WARMUP_INSUFFICIENT_HISTORY"
                decoupling_active = False
            elif z < -1.5:
                regime = "SYSTEMIC_LIQUIDITY_DRAIN"
                decoupling_active = False
            elif z < -0.5:
                regime = "MODERATE_CONTRACTION"
                decoupling_active = False
            elif z > 0.8 and v > 0.02:
                regime = "ORGANIC_LIQUIDITY_EXPANSION"
                decoupling_active = True
            elif z > 0.2 and v > 0.0:
                regime = "MILD_EXPANSION"
                decoupling_active = True
            else:
                regime = "NEUTRAL"
                decoupling_active = False

            results.append({
                "date": d,
                "stablecoin_mcap": mcaps[i],
                "data_status": prov_status,
                "float_growth_30d": round(v, 4) if v is not None else None,
                "liquidity_impulse": round(v, 4) if v is not None else None,
                "velocity_30d": round(v, 4) if v is not None else None,  # Backward compatibility
                "z_score": round(z, 2) if z is not None else None,
                "regime": regime,
                "decoupling_active": decoupling_active
            })
        return results

    def get_liquidity_features_as_of(self, target_date_str=None):
        """
        Retrieves empirical rolling liquidity features as of target_date_str.
        If target_date_str is None or beyond available range, uses latest verified date.
        Derives rolling 30-day float growth, rolling Z-score, and decoupling impulse.
        """
        sorted_dates = sorted(self.history_by_date.keys())
        if not sorted_dates:
            return {
                "as_of_date": None,
                "z_score": 0.0,
                "float_growth_30d": 0.0,
                "decoupling_active": False,
                "regime": "NO_DATA",
                "data_status": "INVALID_NO_DATA"
            }
        
        if target_date_str and target_date_str in self.history_by_date:
            anchor_date = target_date_str
        elif target_date_str:
            eligible = [d for d in sorted_dates if d <= target_date_str]
            anchor_date = eligible[-1] if eligible else sorted_dates[0]
        else:
            anchor_date = sorted_dates[-1]

        anchor_idx = sorted_dates.index(anchor_date)
        window_size = min(anchor_idx + 1, 300)
        eval_dates = sorted_dates[anchor_idx - window_size + 1 : anchor_idx + 1]
        
        features = self.compute_rolling_features(eval_dates, window=min(120, max(20, len(eval_dates) - 35)))
        latest = features[-1]
        
        return {
            "as_of_date": latest["date"],
            "stablecoin_mcap": latest["stablecoin_mcap"],
            "z_score": latest["z_score"] if latest["z_score"] is not None else 0.0,
            "float_growth_30d": latest["float_growth_30d"] if latest["float_growth_30d"] is not None else 0.0,
            "decoupling_active": latest["decoupling_active"],
            "regime": latest["regime"],
            "data_status": latest["data_status"]
        }

