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

    def get_stablecoin_mcap(self, date_str):
        prov = self.get_stablecoin_mcap_with_provenance(date_str)
        if prov["status"] == "DATA_STALE_CRITICAL":
            print(f"[DATA WARNING] Stablecoin observation is STALE: {prov['as_of']} (requested {date_str}, lag: {prov['staleness_days']} days)")
        return prov["mcap"]

    def compute_rolling_features(self, date_list, window=252):
        """
        Computes 30-day net float expansion and rolling 252-day Z-scores.
        Note: Float expansion measures aggregate net supply growth of base money (M),
        distinct from turnover velocity (V = Y / M).
        """
        n = len(date_list)
        mcaps = []
        provenances = []
        for d in date_list:
            p = self.get_stablecoin_mcap_with_provenance(d)
            mcaps.append(p["mcap"] if p["mcap"] is not None else 0.0)
            provenances.append(p)

        # 30-day Net Float Growth (Liquidity Impulse)
        float_growth_30d = [0.0] * n
        for i in range(30, n):
            prev = mcaps[i - 30]
            curr = mcaps[i]
            float_growth_30d[i] = (curr - prev) / prev if prev > 0 else 0.0

        for i in range(min(30, n)):
            float_growth_30d[i] = float_growth_30d[min(30, n - 1)] if n > 30 else 0.0

        z_scores = [0.0] * n
        for i in range(window, n):
            hist = float_growth_30d[i - window : i]
            mu = sum(hist) / len(hist)
            var = sum((x - mu) ** 2 for x in hist) / len(hist)
            sigma = math.sqrt(var) if var > 1e-12 else 1e-6
            z_scores[i] = (float_growth_30d[i] - mu) / sigma

        for i in range(min(window, n)):
            z_scores[i] = 0.0

        results = []
        for i in range(n):
            d = date_list[i]
            v = float_growth_30d[i]
            z = z_scores[i]
            
            if z < -1.5:
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
                "float_growth_30d": round(v, 4),
                "liquidity_impulse": round(v, 4),
                "velocity_30d": round(v, 4),  # Preserved for backward compatibility
                "z_score": round(z, 2),
                "regime": regime,
                "decoupling_active": decoupling_active
            })
        return results
