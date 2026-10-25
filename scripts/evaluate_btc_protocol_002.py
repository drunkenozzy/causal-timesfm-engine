"""Reproducible Confirmatory Evaluator for CTE_CONFIRMATORY_PROTOCOL_002 (BTC-USD).

Scores the preregistered 30-origin holdout window (observations 2026-09-20..2026-10-20,
origins 2026-09-20..2026-10-19, targets 2026-09-21..2026-10-20, h=1) using genuine
rolling-origin neural TimesFM 2.0 500M inference at every origin, plus the as-coded
legacy calibration.py tournament for full audit transparency.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yfinance as yf

ENGINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENGINE_ROOT))

from core.calibration import (
    compute_empirical_quantile_coverage,
    compute_winkler_interval_score,
    run_baseline_tournament,
)
from core.econometrics import EconometricFilter
from core.engine_timesfm import TimesFmBaselineEngine
from core.markov_regime import InstitutionalMarkovEngine
from core.pipeline import CausalTimesFmPipeline

SNAPSHOT_CSV = ENGINE_ROOT / "data" / "btc_protocol_002_snapshot.csv"


def sha256_lf(path: Path) -> str:
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_local_checkpoint_metadata() -> dict:
    hf_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--google--timesfm-2.0-500m-pytorch"
    )
    ref_main = hf_dir / "refs" / "main"
    if ref_main.exists():
        rev = ref_main.read_text(encoding="utf-8").strip()
        snap_dir = hf_dir / "snapshots" / rev
    else:
        snaps = sorted((hf_dir / "snapshots").iterdir()) if (hf_dir / "snapshots").exists() else []
        if not snaps:
            raise RuntimeError("No local HuggingFace snapshot found for google/timesfm-2.0-500m-pytorch")
        snap_dir = snaps[-1]
        rev = snap_dir.name

    ckpt = snap_dir / "torch_model.ckpt"
    if not ckpt.exists():
        raise RuntimeError(f"Checkpoint file missing at {ckpt}")
    return {
        "hf_revision": rev,
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
    }


def load_or_fetch_btc_snapshot(refresh_from_yfinance: bool = False) -> pd.Series:
    if SNAPSHOT_CSV.exists() and not refresh_from_yfinance:
        df = pd.read_csv(SNAPSHOT_CSV)
        s = pd.Series(df["Close"].astype(float).values, index=df["Date"].astype(str).values)
        return s[~s.index.duplicated(keep="last")].sort_index()

    df = yf.download("BTC-USD", start="2026-01-01", end="2026-10-22", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        close_series = df["Close"]["BTC-USD"].dropna()
    else:
        close_series = df["Close"].dropna()

    close_series.index = pd.to_datetime(close_series.index).strftime("%Y-%m-%d")
    close_series = close_series[~close_series.index.duplicated(keep="last")].sort_index()
    close_series = close_series[close_series.index <= "2026-10-20"]

    out_df = pd.DataFrame({"Date": close_series.index, "Close": [float(x) for x in close_series.values]})
    SNAPSHOT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(SNAPSHOT_CSV, index=False, float_format="%.8f", lineterminator="\n")
    return close_series


def bind_engine_to_exact_checkpoint(tfm_engine: TimesFmBaselineEngine, ckpt_path: str):
    import timesfm

    tfm_engine._model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend=tfm_engine.backend,
            per_core_batch_size=1,
            horizon_len=90,
            num_layers=50,
            context_len=2048,
            use_positional_embedding=False,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(path=ckpt_path),
    )
    tfm_engine._initialized = True
    tfm_engine._init_failure_reason = None


def main():
    manifest_path = ENGINE_ROOT / 'docs/PROTOCOL_002_LEGACY_MANIFEST.json'
    frozen = json.loads(manifest_path.read_text(encoding='utf-8'))['sha256_lf']
    for name, expected in frozen.items():
        if sha256_lf(ENGINE_ROOT / name.replace('\\', '/')) != expected:
            raise RuntimeError('ARCHIVED_EVALUATION: restore legacy sources and frozen inputs in an isolated checkout; do not overwrite Protocol 002 with repaired models: ' + name)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-from-yfinance", action="store_true", help="Re-fetch BTC-USD from Yahoo Finance")
    args = parser.parse_args()

    reg_path = ENGINE_ROOT / "CTE_CONFIRMATORY_HOLDOUT_002_REGISTRATION.json"
    proto_path = ENGINE_ROOT / "CTE_CONFIRMATORY_PROTOCOL_002.json"
    manifest_path = ENGINE_ROOT / "CTE_RUNTIME_MANIFEST_R3.4.json"
    term_path = ENGINE_ROOT / "CTE_CONFIRMATORY_PROTOCOL_001_TERMINATION.json"

    reg = json.loads(reg_path.read_text(encoding="utf-8-sig"))
    reg_sealed = json.loads(
        (ENGINE_ROOT / "CTE_CONFIRMATORY_HOLDOUT_002_REGISTRATION_SEALED.json").read_text(encoding="utf-8-sig")
    )

    seal_checks = {
        "registration_sha256": {
            "actual": sha256_lf(reg_path),
            "expected": reg_sealed["sha256"],
        },
        "protocol_sha256": {
            "actual": sha256_lf(proto_path),
            "expected": reg["bindings"]["protocol_sha256"],
        },
        "runtime_manifest_sha256": {
            "actual": sha256_lf(manifest_path),
            "expected": reg["bindings"]["runtime_manifest_sha256"],
        },
        "protocol_001_termination_sha256": {
            "actual": sha256_lf(term_path),
            "expected": reg["bindings"]["protocol_001_termination_sha256"],
        },
    }
    for k, v in seal_checks.items():
        v["verified"] = (v["actual"] == v["expected"])
        if not v["verified"]:
            raise RuntimeError(f"SEAL_VERIFICATION_FAILED for {k}: actual={v['actual']} expected={v['expected']}")

    now_utc = datetime.now(timezone.utc)
    earliest_utc = datetime.fromisoformat(reg["earliest_permitted_execution_utc"].replace("Z", "+00:00"))
    if now_utc < earliest_utc:
        raise RuntimeError("Holdout window has not yet reached earliest_permitted_execution_utc!")

    close_series = load_or_fetch_btc_snapshot(refresh_from_yfinance=args.refresh_from_yfinance)
    dates = list(close_series.index)
    values = [float(x) for x in close_series.values]
    input_snapshot_sha256 = sha256_lf(SNAPSHOT_CSV)

    holdout_obs = close_series[(close_series.index >= "2026-09-20") & (close_series.index <= "2026-10-20")]
    if len(holdout_obs) != 31:
        raise RuntimeError(f"Expected 31 daily closes in holdout window, got {len(holdout_obs)}")

    origin_indices = [i for i, d in enumerate(dates) if "2026-09-20" <= d <= "2026-10-19"]
    if len(origin_indices) != 30:
        raise RuntimeError(f"Expected 30 confirmatory origins, got {len(origin_indices)}")

    ckpt_meta = resolve_local_checkpoint_metadata()
    tfm_engine = TimesFmBaselineEngine()
    bind_engine_to_exact_checkpoint(tfm_engine, ckpt_meta["checkpoint_path"])

    pipeline = CausalTimesFmPipeline()
    pipeline.tfm_engine = tfm_engine
    pipeline.ledger = SimpleNamespace(record_forecast=lambda **kwargs: {"forecast_id": "CONFIRMATORY_EVAL"})

    first_origin_idx = origin_indices[0]
    warmup_start_idx = max(30, first_origin_idx - 15)
    all_eval_indices = list(range(warmup_start_idx, origin_indices[-1] + 1))

    rolling_records = {}
    for idx in all_eval_indices:
        origin_date = dates[idx]
        target_date = dates[idx + 1]
        hist_slice = values[: idx + 1]
        actual_next = values[idx + 1]

        tfm_prior = tfm_engine.forecast(hist_slice, horizon_days=1)
        if not tfm_prior.get("timesfm_executed", False) or tfm_prior.get("degraded_mode", True):
            raise RuntimeError(f"NOT_EVALUATED_RUNTIME_FAILURE at {origin_date}: {tfm_prior}")

        returns = [(hist_slice[i] - hist_slice[i - 1]) / hist_slice[i - 1] for i in range(1, len(hist_slice))]
        mean_ret = sum(returns) / len(returns)
        period_vol = math.sqrt(sum((r - mean_ret) ** 2 for r in returns) / max(1, len(returns) - 1))
        ret_t = returns[-1]

        w_trend = min(60, len(hist_slice))
        trailing_window = hist_slice[-w_trend:]
        mean_tr = sum(trailing_window) / len(trailing_window)
        std_tr = math.sqrt(sum((x - mean_tr) ** 2 for x in trailing_window) / max(1, len(trailing_window) - 1))
        z_trend_val = (hist_slice[-1] - mean_tr) / (std_tr if std_tr > 1e-6 else 1.0)
        z_trend = max(-3.5, min(3.5, z_trend_val))

        liq_features = pipeline.liq_engine.get_liquidity_features_as_of(origin_date)
        z_liq = liq_features["z_score"]
        decoupling_active = liq_features["decoupling_active"]

        struct_val = pipeline.struct_engine.compute_structural_target(
            current_price=hist_slice[-1],
            macro_liquidity_regime="expanding" if decoupling_active else "neutral",
            circulating_supply=0.92,
            total_supply=1.00,
            minsky_stage="Hedge" if decoupling_active else "Speculative",
        )

        markov = InstitutionalMarkovEngine(period_volatility=period_vol, observation_frequency="D")
        state = markov.update(step_ret=ret_t, z_driver=z_liq, z_trend=z_trend, decoupling_active=decoupling_active)

        cond_markov_only = markov.condition_timesfm_quantiles(
            tfm_p10=tfm_prior["p10_downside"],
            tfm_p50=tfm_prior["p50_expected"],
            tfm_p90=tfm_prior["p90_upside"],
            forward_xi=state["state_vector"],
            asset_vol_scale=period_vol,
            horizon_steps=1,
            transition_matrix=state["transition_matrix"],
            horizon_mode="frozen_transition",
            structural_target=None,
        )

        cond_full_pipeline = markov.condition_timesfm_quantiles(
            tfm_p10=tfm_prior["p10_downside"],
            tfm_p50=tfm_prior["p50_expected"],
            tfm_p90=tfm_prior["p90_upside"],
            forward_xi=state["state_vector"],
            asset_vol_scale=period_vol,
            horizon_steps=1,
            transition_matrix=state["transition_matrix"],
            horizon_mode="frozen_transition",
            structural_target=struct_val["structural_target"],
        )

        p_m0 = round(hist_slice[-1], 4)
        drift = (hist_slice[-1] / hist_slice[0]) ** (1.0 / len(hist_slice))
        p_m0b = round(hist_slice[-1] * drift, 4)

        w_mom = min(5, len(hist_slice) - 1)
        mom_5 = (hist_slice[-1] / hist_slice[-1 - w_mom]) ** (1.0 / w_mom)
        v_struct_damped = round(hist_slice[-1] * (mom_5 ** 0.8), 4)

        m4_p10 = round(cond_full_pipeline["downside_floor"], 4)
        m4_p50 = round(cond_full_pipeline["expected_target"], 4)
        m4_p90 = round(cond_full_pipeline["upside_ceiling"], 4)

        rolling_records[idx] = {
            "origin_idx": idx,
            "origin_date": origin_date,
            "target_date": target_date,
            "history_rows": len(hist_slice),
            "history_sha256": hashlib.sha256(np.asarray(hist_slice, dtype=np.float64).tobytes()).hexdigest(),
            "origin_close": p_m0,
            "actual_close": round(actual_next, 4),
            "m0_persistence": p_m0,
            "m0b_drift": p_m0b,
            "m1_timesfm_p10": round(tfm_prior["p10_downside"], 4),
            "m1_timesfm_p50": round(tfm_prior["p50_expected"], 4),
            "m1_timesfm_p90": round(tfm_prior["p90_upside"], 4),
            "struct_macro_target": round(struct_val["structural_target"], 4),
            "struct_damped_target": v_struct_damped,
            "m4_pipeline_p10": m4_p10,
            "m4_pipeline_p50": m4_p50,
            "m4_pipeline_p90": m4_p90,
            "m4_pipeline_ordering_valid": bool(m4_p10 <= m4_p50 <= m4_p90),
            "m4_markov_only_p50": round(cond_markov_only["expected_target"], 4),
            "pipeline_alpha": cond_full_pipeline["structural_weight"],
            "effective_regime": state["effective_regime"],
            "fragility_score": state["fragility_score"],
            "liquidity_as_of_date": liq_features["as_of_date"],
            "liquidity_reported_status": liq_features["data_status"],
        }

    confirmatory_rows = []
    for idx in origin_indices:
        prior_indices = [k for k in all_eval_indices if k < idx]
        hist_actual = [rolling_records[k]["actual_close"] for k in prior_indices]
        hist_m1 = [rolling_records[k]["m1_timesfm_p50"] for k in prior_indices]
        hist_struct_macro = [rolling_records[k]["struct_macro_target"] for k in prior_indices]
        hist_struct_damped = [rolling_records[k]["struct_damped_target"] for k in prior_indices]

        alpha_macro_res = InstitutionalMarkovEngine.estimate_optimal_structural_weight(
            hist_actual, hist_m1, hist_struct_macro, min_inner_train=5
        )
        alpha_damped_res = InstitutionalMarkovEngine.estimate_optimal_structural_weight(
            hist_actual, hist_m1, hist_struct_damped, min_inner_train=5
        )

        row = dict(rolling_records[idx])
        a_macro = alpha_macro_res["alpha_star"]
        a_damped = alpha_damped_res["alpha_star"]
        row["alpha_nested_macro"] = a_macro
        row["alpha_nested_damped"] = a_damped
        row["m4_nested_macro_p50"] = round(
            (1.0 - a_macro) * row["m1_timesfm_p50"] + a_macro * row["struct_macro_target"], 4
        )
        row["m4_nested_damped_p50"] = round(
            (1.0 - a_damped) * row["m1_timesfm_p50"] + a_damped * row["struct_damped_target"], 4
        )
        confirmatory_rows.append(row)

    ef = EconometricFilter()

    def score_model(pred_key: str):
        errs = [r["actual_close"] - r[pred_key] for r in confirmatory_rows]
        mspe = sum(e ** 2 for e in errs) / len(errs)
        rmse = math.sqrt(mspe)
        mae = sum(abs(e) for e in errs) / len(errs)
        dir_hits = 0
        non_zero_preds = 0
        for r in confirmatory_rows:
            actual_dir = r["actual_close"] - r["origin_close"]
            pred_dir = r[pred_key] - r["origin_close"]
            if abs(pred_dir) > 1e-9:
                non_zero_preds += 1
                if (actual_dir > 0 and pred_dir > 0) or (actual_dir < 0 and pred_dir < 0):
                    dir_hits += 1
        return {
            "errors": errs,
            "mspe": round(mspe, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "directional_hits": dir_hits if non_zero_preds > 0 else None,
            "directional_accuracy": round(dir_hits / len(confirmatory_rows), 4) if non_zero_preds > 0 else None,
        }

    m0_res = score_model("m0_persistence")
    m0b_res = score_model("m0b_drift")
    m1_res = score_model("m1_timesfm_p50")
    m4_pipe_res = score_model("m4_pipeline_p50")
    m4_markov_res = score_model("m4_markov_only_p50")
    m4_nested_macro_res = score_model("m4_nested_macro_p50")
    m4_nested_damped_res = score_model("m4_nested_damped_p50")

    def compare_m4_vs_m1(m4_score):
        cw = ef.compute_clark_west_test(m4_score["errors"], m1_res["errors"], h=1)
        boot_null = ef.block_bootstrap_null_test(m4_score["errors"], m1_res["errors"], h=1, n_boot=1000)
        boot_ci = ef.block_bootstrap_effect_ci(m4_score["errors"], m1_res["errors"], h=1, n_boot=1000)
        dm = ef.compute_diebold_mariano_test(m4_score["errors"], m1_res["errors"], h=1)
        win_m4 = sum(1 for e4, e1 in zip(m4_score["errors"], m1_res["errors"]) if abs(e4) < abs(e1))
        win_m1 = sum(1 for e4, e1 in zip(m4_score["errors"], m1_res["errors"]) if abs(e1) < abs(e4))
        ties = len(m1_res["errors"]) - win_m4 - win_m1
        return {
            "mspe": m4_score["mspe"],
            "rmse": m4_score["rmse"],
            "mae": m4_score["mae"],
            "theils_u2": round(m4_score["rmse"] / m0_res["rmse"], 4),
            "directional_hits": m4_score["directional_hits"],
            "directional_accuracy": m4_score["directional_accuracy"],
            "delta_mspe_vs_m1_pct": round((1.0 - (m4_score["mspe"] / m1_res["mspe"])) * 100.0, 4),
            "delta_rmse_vs_m1_pct": round((1.0 - (m4_score["rmse"] / m1_res["rmse"])) * 100.0, 4),
            "delta_mspe_vs_m0_pct": round((1.0 - (m4_score["mspe"] / m0_res["mspe"])) * 100.0, 4),
            "win_loss_tie_vs_m1": f"{win_m4}W - {win_m1}L - {ties}T",
            "clark_west": cw,
            "bootstrap_null": boot_null,
            "bootstrap_effect_ci": boot_ci,
            "diebold_mariano": dm,
            "h1_confirmed": bool(m4_score["mspe"] < m1_res["mspe"] and cw["p_value"] < 0.05),
        }

    actuals_list = [r["actual_close"] for r in confirmatory_rows]
    m1_cov = compute_empirical_quantile_coverage(
        actuals_list,
        p10_series=[r["m1_timesfm_p10"] for r in confirmatory_rows],
        p50_series=[r["m1_timesfm_p50"] for r in confirmatory_rows],
        p90_series=[r["m1_timesfm_p90"] for r in confirmatory_rows],
    )
    m4_cov = compute_empirical_quantile_coverage(
        actuals_list,
        p10_series=[r["m4_pipeline_p10"] for r in confirmatory_rows],
        p50_series=[r["m4_pipeline_p50"] for r in confirmatory_rows],
        p90_series=[r["m4_pipeline_p90"] for r in confirmatory_rows],
    )
    m1_wis = sum(
        compute_winkler_interval_score(r["actual_close"], r["m1_timesfm_p10"], r["m1_timesfm_p90"], alpha=0.20)
        for r in confirmatory_rows
    ) / len(confirmatory_rows)
    m4_wis = sum(
        compute_winkler_interval_score(r["actual_close"], r["m4_pipeline_p10"], r["m4_pipeline_p90"], alpha=0.20)
        for r in confirmatory_rows
    ) / len(confirmatory_rows)

    slice_60 = values[origin_indices[0] - 29 : origin_indices[-1] + 2]
    as_coded_tourney = run_baseline_tournament(slice_60, h=1, min_train_len=30, enforce_confirmatory_timesfm=True)

    output_report = {
        "protocol_id": "CTE_CONFIRMATORY_PROTOCOL_002",
        "holdout_id": "CTE_CONFIRMATORY_HOLDOUT_002",
        "evaluated_at_utc": now_utc.isoformat(),
        "asset": "BTC-USD",
        "provider": "Yahoo Finance (yfinance)",
        "input_snapshot_file": "data/btc_protocol_002_snapshot.csv",
        "input_context_range": [dates[0], dates[-1]],
        "input_context_rows": len(values),
        "input_snapshot_sha256": input_snapshot_sha256,
        "observations_range_inclusive": ["2026-09-20", "2026-10-20"],
        "confirmatory_origins_range": [confirmatory_rows[0]["origin_date"], confirmatory_rows[-1]["origin_date"]],
        "target_dates_range": [confirmatory_rows[0]["target_date"], confirmatory_rows[-1]["target_date"]],
        "n_origins": len(confirmatory_rows),
        "runtime_provenance": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                p: importlib.metadata.version(p)
                for p in ["timesfm", "torch", "numpy", "pandas", "scipy", "yfinance"]
            },
            "checkpoint_repository": "google/timesfm-2.0-500m-pytorch",
            "huggingface_snapshot_revision": ckpt_meta["hf_revision"],
            "checkpoint_path_loaded": ckpt_meta["checkpoint_path"],
            "checkpoint_sha256": ckpt_meta["checkpoint_sha256"],
        },
        "seal_verification": seal_checks,
        "baselines": {
            "M0_Persistence": {
                "mspe": m0_res["mspe"],
                "rmse": m0_res["rmse"],
                "mae": m0_res["mae"],
                "theils_u2": 1.0,
                "directional_hits": None,
                "directional_accuracy": None,
                "directional_note": "Persistence predicts zero movement (origin_close == forecast).",
            },
            "M0b_Drift": {
                "mspe": m0b_res["mspe"],
                "rmse": m0b_res["rmse"],
                "mae": m0b_res["mae"],
                "theils_u2": round(m0b_res["rmse"] / m0_res["rmse"], 4),
                "directional_hits": m0b_res["directional_hits"],
                "directional_accuracy": m0b_res["directional_accuracy"],
            },
            "M1_TIMESFM": {
                "mspe": m1_res["mspe"],
                "rmse": m1_res["rmse"],
                "mae": m1_res["mae"],
                "theils_u2": round(m1_res["rmse"] / m0_res["rmse"], 4),
                "directional_hits": m1_res["directional_hits"],
                "directional_accuracy": m1_res["directional_accuracy"],
                "delta_mspe_vs_m0_pct": round((1.0 - m1_res["mspe"] / m0_res["mspe"]) * 100.0, 4),
                "quantile_coverage": m1_cov,
                "mean_winkler_score_80pct": round(m1_wis, 4),
                "ordering_violations": sum(
                    1
                    for r in confirmatory_rows
                    if not (r["m1_timesfm_p10"] <= r["m1_timesfm_p50"] <= r["m1_timesfm_p90"])
                ),
            },
        },
        "m4_evaluations": {
            "M4_Pipeline_ShadowV2_Mechanism": compare_m4_vs_m1(m4_pipe_res),
            "M4_Markov_Conditioned_Only": compare_m4_vs_m1(m4_markov_res),
            "M4_NestedCV_MacroAnchor": compare_m4_vs_m1(m4_nested_macro_res),
            "M4_NestedCV_DampedMomentum": compare_m4_vs_m1(m4_nested_damped_res),
        },
        "m4_pipeline_quantile_calibration": {
            "quantile_coverage": m4_cov,
            "mean_winkler_score_80pct": round(m4_wis, 4),
            "ordering_violations": sum(1 for r in confirmatory_rows if not r["m4_pipeline_ordering_valid"]),
        },
        "as_coded_calibration_tournament": as_coded_tourney,
        "daily_confirmatory_ledger": confirmatory_rows,
    }

    out_path = ENGINE_ROOT / "docs" / "CTE_CONFIRMATORY_HOLDOUT_002_RESULTS.json"
    out_path.write_text(json.dumps(output_report, indent=2), encoding="utf-8")
    print(f"Saved official confirmatory results to {out_path}")


if __name__ == "__main__":
    main()
