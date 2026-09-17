"""
Engine 1: Statistical Foundation Model Baseline (Google TimesFM)
==============================================================
Provides the empirical, non-linear pattern recognition baseline.
Works natively with pure Python standard library and accelerates with TimesFM/NumPy if installed.
"""

import math

class TimesFmBaselineEngine:
    def __init__(self, repo_id="google/timesfm-2.0-500m-pytorch", backend="cpu"):
        self.repo_id = repo_id
        self.backend = backend
        self._model = None
        self._initialized = False

    def _try_init_timesfm(self):
        """Attempts to load official TimesFM PyTorch weights if installed."""
        if self._initialized:
            return self._model is not None
        try:
            import timesfm
            self._model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend=self.backend,
                    per_core_batch_size=1,
                    horizon_len=90
                ),
                checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=self.repo_id)
            )
            self._initialized = True
            return True
        except Exception:
            self._initialized = True
            self._model = None
            return False

    def detect_volatility_squeeze(self, series, short_window=20, long_window=60):
        """Detects volatility compression below historical bandwidth mean."""
        vals = [float(x) for x in series]
        if len(vals) < long_window:
            return False, 0.0

        recent = vals[-short_window:]
        hist = vals[-long_window:]

        m_rec = sum(recent) / len(recent)
        std_rec = math.sqrt(sum((x - m_rec)**2 for x in recent) / len(recent))

        m_hist = sum(hist) / len(hist)
        std_hist = math.sqrt(sum((x - m_hist)**2 for x in hist) / len(hist))

        bw_recent = (std_rec / (m_rec + 1e-9)) * 100
        bw_hist = (std_hist / (m_hist + 1e-9)) * 100

        is_squeezed = bw_recent < (bw_hist * 0.70)
        return is_squeezed, round(float(bw_recent), 2)

    def forecast(self, history_series, horizon_days=30, freq_indicator=0):
        """Generates probabilistic quantile corridors (P10, P50, P90)."""
        vals = [float(x) for x in history_series]
        n = len(vals)
        current_price = vals[-1]

        is_squeezed, bandwidth = self.detect_volatility_squeeze(vals)

        # 1. Official TimesFM if loaded
        if self._try_init_timesfm():
            try:
                point_forecast, quantile_forecast = self._model.forecast(
                    inputs=[vals],
                    freq=[freq_indicator]
                )
                p50 = float(point_forecast[0][horizon_days - 1])
                p10 = float(quantile_forecast[0, horizon_days - 1, 1])
                p90 = float(quantile_forecast[0, horizon_days - 1, 9])
                return {
                    "engine": "TimesFM-Neural",
                    "current_price": current_price,
                    "horizon_days": horizon_days,
                    "p10_downside": round(p10, 4),
                    "p50_expected": round(p50, 4),
                    "p90_upside": round(p90, 4),
                    "expected_return_pct": round(((p50 / current_price) - 1.0) * 100, 2),
                    "upside_tail_pct": round(((p90 / current_price) - 1.0) * 100, 2),
                    "downside_risk_pct": round(((p10 / current_price) - 1.0) * 100, 2),
                    "volatility_squeeze": is_squeezed,
                    "bandwidth": bandwidth
                }
            except Exception:
                pass

        # 2. Geometric Quantile Autoregressive Engine
        log_diffs = [math.log(vals[i]) - math.log(vals[i - 1]) for i in range(1, n)]
        mean_diff = sum(log_diffs) / len(log_diffs)
        daily_vol = math.sqrt(sum((r - mean_diff)**2 for r in log_diffs) / len(log_diffs))

        lookback = min(30, n)
        trend_drift = (vals[-1] / vals[-lookback]) - 1.0
        damped_drift = trend_drift * (0.35 if horizon_days > 30 else 0.50)

        p50 = current_price * (1.0 + damped_drift)

        horizon_vol = daily_vol * math.sqrt(horizon_days)
        z_score = 1.645
        squeeze_multiplier = 1.4 if is_squeezed else 1.0

        p90 = p50 * math.exp(z_score * horizon_vol * squeeze_multiplier)
        p10 = p50 * math.exp(-z_score * horizon_vol)

        return {
            "engine": "TimesFM-Quantile-Statistical",
            "current_price": current_price,
            "horizon_days": horizon_days,
            "p10_downside": round(float(p10), 4),
            "p50_expected": round(float(p50), 4),
            "p90_upside": round(float(p90), 4),
            "expected_return_pct": round(((p50 / current_price) - 1.0) * 100, 2),
            "upside_tail_pct": round(((p90 / current_price) - 1.0) * 100, 2),
            "downside_risk_pct": round(((p10 / current_price) - 1.0) * 100, 2),
            "volatility_squeeze": is_squeezed,
            "bandwidth": bandwidth
        }
