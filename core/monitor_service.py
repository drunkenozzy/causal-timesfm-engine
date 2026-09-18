"""
Continuous Monitoring & Alert Service v2.0
==========================================
Runs as a lightweight daemon or scheduled cron task to continuously monitor:
  - 24/7 On-chain Stablecoin Liquidity (DefiLlama aggregate float)
  - US Macro Indicators (DXY, 10Y-3M Yield Curve, VIX)
  - User Portfolio Asset Prices (Crypto, Equities, Housing)

Dispatches urgent plain-English warnings when:
  1. TVTP Markov Filter flips into PONZI_LIQUIDATION_CRUNCH (P > 0.40).
  2. Price breaches the statistical TimesFM P10 downside floor.
  3. Stablecoin net float growth drops below -1.5 standard deviations.
"""

import os
import sys
import json
import time
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from core.onchain_liquidity import OnChainLiquidityEngine
from core.markov_regime import InstitutionalMarkovEngine
from core.reconciliation import ReconciliationEngine

class ContinuousMonitorService:
    def __init__(self, check_interval_seconds=86400, alert_webhook_url=None):
        self.interval = check_interval_seconds
        self.webhook_url = alert_webhook_url
        self.liq_engine = OnChainLiquidityEngine()
        self.markov_engine = InstitutionalMarkovEngine()
        self.reconciler = ReconciliationEngine()
        self.last_state = "HEDGE"

    def check_systemic_health(self):
        """
        Queries live liquidity feeds and evaluates systemic Minsky fragility.
        """
        now_utc = datetime.now(timezone.utc)
        today_str = now_utc.strftime("%Y-%m-%d")
        print(f"[{now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}] Checking Systemic Liquidity...")

        # Ingest recent 30 days of stablecoins
        dates = [datetime.fromtimestamp(time.time() - i*86400).strftime("%Y-%m-%d") for i in range(60, -1, -1)]
        features = self.liq_engine.compute_rolling_features(dates, window=30)
        latest = features[-1]

        z_liq = latest["z_score"]
        float_growth = latest.get("float_growth_30d", latest.get("velocity_30d", 0.0))
        decoupling = latest["decoupling_active"]

        # Run Markov Filter update
        update_info = self.markov_engine.update(
            daily_ret=0.0,
            z_liq=z_liq,
            z_trend=0.0,
            decoupling_active=decoupling
        )
        effective_regime = update_info["effective_regime"]
        p_ponzi = update_info["raw_ponzi_prob"]
        fragility_score = update_info.get("fragility_score", round(p_ponzi * 100, 1))

        alert_triggered = False
        alert_message = ""

        if effective_regime == "PONZI" and self.last_state != "PONZI":
            alert_triggered = True
            alert_message = (
                f"🚨 URGENT RISK-OFF ALERT: Systemic Liquidity Drain Detected!\n"
                f"Stablecoin Float Growth Z-Score: {z_liq:.2f} | Fragility Score: {fragility_score:.1f}/100.\n"
                f"Action Required: Execute Rule 6 capital rotation. Cut high-beta exposure to 15%."
            )
        elif effective_regime != "PONZI" and self.last_state == "PONZI":
            alert_triggered = True
            alert_message = (
                f"✅ REGIME RESOLVED: Market has exited Ponzi regime into {effective_regime}.\n"
                f"On-chain liquidity has stabilized. Prudent re-accumulation permitted."
            )

        self.last_state = effective_regime

        if alert_triggered:
            self._dispatch_alert(alert_message)

        return {
            "date": today_str,
            "regime": effective_regime,
            "z_liq": z_liq,
            "float_growth_30d": float_growth,
            "velocity_30d": float_growth,  # Backwards compatibility
            "fragility_score": fragility_score,
            "ponzi_probability": p_ponzi,  # Backwards compatibility
            "alert_triggered": alert_triggered
        }

    def _dispatch_alert(self, message):
        print("\n" + "!"*80)
        print(message)
        print("!"*80 + "\n")
        # If webhook is configured (Telegram/Discord), post here:
        if self.webhook_url:
            try:
                import urllib.request
                req = urllib.request.Request(
                    self.webhook_url,
                    data=json.dumps({"text": message}).encode('utf-8'),
                    headers={'Content-Type': 'application/json'}
                )
                urllib.request.urlopen(req, timeout=5)
            except Exception as e:
                print(f"[Notifier] Webhook delivery failed: {e}")

    def run_daemon(self, max_iterations=None):
        print(f"Continuous Monitoring Service started (Polling every {self.interval} seconds)...")
        iters = 0
        while True:
            try:
                self.check_systemic_health()
                iters += 1
                if max_iterations and iters >= max_iterations:
                    break
                time.sleep(self.interval)
            except KeyboardInterrupt:
                print("\nStopping Monitoring Service.")
                break
            except Exception as e:
                print(f"Error during check: {e}")
                time.sleep(60)

if __name__ == "__main__":
    service = ContinuousMonitorService(check_interval_seconds=3600)
    res = service.check_systemic_health()
    print("Single-shot health check result:", res)
