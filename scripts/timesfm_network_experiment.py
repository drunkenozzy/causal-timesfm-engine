"""Three-way retrospective neural experiment; isolated from BTC Protocol 002.

Fit identical ridge residual corrections to past rolling-origin TimesFM errors.
Never use fitted/context reconstructions as historical out-of-sample forecasts.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd

from scripts.network_experiment import GRAPH_COLUMNS, changes, feature_frame, ridge_prediction, validate_panel

CONFIG = {
    'experiment': 'TIMESFM_NETWORK_COMPARISON_001',
    'targets': ['ETH', 'Brent', 'TRY'],
    'window': 60, 'short_window': 20, 'min_train': 60, 'train_window': 252, 'ridge_penalty': 10.,
    'bootstrap_replications': 1000, 'bootstrap_block_length': 5, 'seed': 42,
    'primary_metric': 'next_joint_session_log_return_MSPE', 'secondary_metric': 'price_MSPE',
    'arms': ['timesfm', 'timesfm_raw', 'timesfm_raw_network'],
    'horizon': 'next joint observed session; irregular calendar spacing',
    'evaluation_type': 'RETROSPECTIVE_EXPLORATORY',
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class NeuralBaseline:
    """Pinned local 500M checkpoint, canonical architecture, no fallback."""
    def __init__(self, checkpoint_path):
        import torch
        from timesfm import TimesFmHparams, TimesFmCheckpoint
        from timesfm.timesfm_torch import TimesFmTorch
        torch.set_num_threads(4)
        torch.manual_seed(CONFIG['seed'])
        hparams = TimesFmHparams(backend='cpu', per_core_batch_size=1, horizon_len=90,
                                num_layers=50, context_len=2048, use_positional_embedding=False,
                                point_forecast_mode='median')
        self.model = TimesFmTorch(hparams=hparams, checkpoint=TimesFmCheckpoint(path=str(checkpoint_path)))
        self.provenance = {
            'engine': 'M1_TIMESFM', 'timesfm_executed': True, 'degraded_mode': False,
            'checkpoint_repository': 'google/timesfm-2.0-500m-pytorch',
            'checkpoint_sha256': sha256_file(checkpoint_path), 'hparams': asdict(hparams),
            'torch_threads': 4, 'python': platform.python_version(),
            'packages': {p: importlib.metadata.version(p) for p in ['timesfm', 'torch', 'numpy', 'pandas']},
        }

    def predict(self, history):
        point, _ = self.model.forecast(inputs=[history], freq=[0])
        value = float(point[0, 0])
        if not np.isfinite(value) or value <= 0:
            raise RuntimeError('INVALID_NEURAL_FORECAST: no fallback or silently dropped origins')
        return value


def neural_history(panel, target, predict, window=60, on_forecast=None):
    panel = validate_panel(panel)
    if target not in CONFIG['targets'] or target not in panel:
        raise ValueError('Unsupported target; BTC is excluded')
    records = []
    for i in range(window, len(panel) - 1):
        context = panel[target].iloc[:i + 1].to_numpy(copy=True)
        prediction = float(predict(context))
        if not np.isfinite(prediction) or prediction <= 0:
            raise RuntimeError('Invalid baseline prediction')
        record = {'origin': panel.index[i].strftime('%Y-%m-%d'),
                  'target': panel.index[i + 1].strftime('%Y-%m-%d'),
                  'prediction': prediction, 'history_rows': len(context),
                  'history_sha256': hashlib.sha256(context.tobytes()).hexdigest()}
        records.append(record)
        if on_forecast:
            on_forecast(record)
    return records


def paired_skill_interval(reference, candidate, reps=1000, block=5, seed=42):
    """Descriptive moving-block bootstrap interval; not a confirmatory decision."""
    ref, cand = np.asarray(reference), np.asarray(candidate)
    if len(ref) < block or ref.mean() <= 0:
        return None
    rng = np.random.default_rng(seed)
    skills = []
    for _ in range(reps):
        starts = rng.integers(0, len(ref) - block + 1, size=int(np.ceil(len(ref) / block)))
        selected = np.concatenate([np.arange(s, s + block) for s in starts])[:len(ref)]
        denominator = ref[selected].mean()
        if denominator > 0:
            skills.append(100 * (1 - cand[selected].mean() / denominator))
    return np.percentile(skills, [2.5, 97.5]).tolist() if skills else None


def compare(panel, target, baseline_records, config=None):
    settings = dict(CONFIG if config is None else config)
    panel = validate_panel(panel)
    if target not in settings['targets'] or target not in panel:
        raise ValueError('Unsupported target')
    window, min_train = settings['window'], settings['min_train']
    if min_train < 20 or settings['train_window'] < min_train or settings['ridge_penalty'] <= 0:
        raise ValueError('Invalid residual training configuration')
    expected = panel.index[window:-1]
    origins = [r['origin'] for r in baseline_records]
    if origins != expected.strftime('%Y-%m-%d').tolist():
        raise ValueError('Baseline must contain exactly the ordered rolling origins; no missing/duplicate rows')
    for i, record in enumerate(baseline_records, start=window):
        if record['target'] != panel.index[i + 1].strftime('%Y-%m-%d'):
            raise ValueError('Baseline target alignment mismatch')
        context = panel[target].iloc[:i + 1].to_numpy(copy=True)
        if record['history_rows'] != len(context) or record['history_sha256'] != hashlib.sha256(context.tobytes()).hexdigest():
            raise ValueError('Baseline history hash mismatch')
    baseline_price = np.asarray([r['prediction'] for r in baseline_records])
    if not np.isfinite(baseline_price).all() or (baseline_price <= 0).any():
        raise ValueError('Invalid neural predictions')
    features = feature_frame(panel, target, window=window, short_window=settings['short_window']).loc[expected]
    actual = changes(panel)[target].shift(-1).loc[expected].to_numpy()
    current_price = panel.loc[expected, target].to_numpy()
    neural_return = np.log(baseline_price / current_price)
    residual = actual - neural_return
    raw = [c for c in features if c.startswith('change_')] + ['target_volatility']
    groups = {'timesfm_raw': raw, 'timesfm_raw_network': raw + GRAPH_COLUMNS}
    rows = []
    for i in range(min_train, len(features)):
        start = max(0, i - settings['train_window'])
        predictions = {'timesfm': float(neural_return[i])}
        for name, columns in groups.items():
            correction = ridge_prediction(features[columns].iloc[start:i].to_numpy(), residual[start:i],
                                          features[columns].iloc[i].to_numpy(), settings['ridge_penalty'])
            predictions[name] = float(neural_return[i] + correction)
        rows.append({'origin': baseline_records[i]['origin'], 'target': baseline_records[i]['target'],
                     'actual_log_return': float(actual[i]), 'current_price': float(current_price[i]),
                     'predictions_log_return': predictions,
                     'training_origins': i - start, 'last_training_target': baseline_records[i - 1]['target']})
    losses = {arm: [(r['actual_log_return'] - r['predictions_log_return'][arm]) ** 2 for r in rows]
              for arm in settings['arms']}
    price_losses = {arm: [(r['current_price'] * (np.exp(r['actual_log_return']) - np.exp(r['predictions_log_return'][arm]))) ** 2
                         for r in rows] for arm in settings['arms']}
    scores = {arm: float(np.mean(values)) for arm, values in losses.items()} if rows else {}
    comparisons = {}
    if rows:
        for ref, candidate in [('timesfm', 'timesfm_raw'), ('timesfm', 'timesfm_raw_network'),
                               ('timesfm_raw', 'timesfm_raw_network')]:
            comparisons[f'{candidate}_vs_{ref}'] = {
                'skill_pct': 100 * (1 - scores[candidate] / scores[ref]) if scores[ref] > 0 else None,
                'descriptive_95pct_block_bootstrap_interval': paired_skill_interval(
                    losses[ref], losses[candidate], settings['bootstrap_replications'],
                    settings['bootstrap_block_length'], settings['seed'])}
    return {'experiment': settings['experiment'], 'evaluation_type': settings['evaluation_type'],
            'status': 'EXPLORATORY' if len(rows) >= 30 else 'INSUFFICIENT_OOS', 'target_asset': target,
            'origins': len(rows), 'primary_metric': settings['primary_metric'], 'mspe': scores,
            'price_mspe': {arm: float(np.mean(values)) for arm, values in price_losses.items()} if rows else {},
            'comparisons': comparisons, 'predictions': rows,
            'limitations': ['Only 32 evaluation origins in the original snapshot; uncertainty is large.',
                           'Uses the already-inspected historical snapshot; no untouched holdout claim.',
                           'Joint-session spacing is irregular and source close times differ.',
                           'Bootstrap intervals are descriptive, not adjusted for multiple experiments.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices', required=True)
    parser.add_argument('--checkpoint', required=True, help='Local torch_model.ckpt, TimesFM 2.0 500M')
    parser.add_argument('--output-dir', required=True, help='New run directory; never overwritten')
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    panel = validate_panel(pd.read_csv(args.prices, index_col='Date', parse_dates=['Date']))
    manifest = {'config': CONFIG, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'input_sha256': sha256_file(args.prices), 'checkpoint_sha256': sha256_file(args.checkpoint),
                'source_sha256': sha256_file(__file__),
                'features_source_sha256': sha256_file(Path(__file__).with_name('network_experiment.py'))}
    try:
        manifest['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        manifest['git_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        manifest['git_commit'], manifest['git_dirty'] = None, None
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    started = time.monotonic()
    try:
        model = NeuralBaseline(args.checkpoint)
        # Synthetic-only smoke test precedes any historical forecast.
        smoke = model.predict(np.linspace(100., 110., 100))
        manifest['runtime'] = model.provenance
        manifest['synthetic_smoke_prediction'] = smoke
        (output / 'runtime.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'TimesFM loaded; synthetic smoke test succeeded ({time.monotonic() - started:.1f}s)', flush=True)
        for asset in CONFIG['targets']:
            with (output / f'baseline_{asset}.jsonl').open('x', encoding='utf-8') as handle:
                count = 0
                def record(row):
                    nonlocal count
                    handle.write(json.dumps(row, allow_nan=False) + '\n')
                    handle.flush()
                    count += 1
                    if count % 10 == 0:
                        print(f'{asset}: {count}/{len(panel) - CONFIG["window"] - 1} neural origins; elapsed {time.monotonic() - started:.0f}s', flush=True)
                records = neural_history(panel, asset, model.predict, CONFIG['window'], record)
            result = compare(panel, asset, records)
            result['timesfm_executed'] = True
            result['degraded_mode'] = False
            (output / f'result_{asset}.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
            print(json.dumps({k: v for k, v in result.items() if k != 'predictions'}), flush=True)
        (output / 'COMPLETED.json').write_text(json.dumps({'elapsed_seconds': time.monotonic() - started}), encoding='utf-8')
    except Exception as exc:
        (output / 'FAILED.json').write_text(json.dumps({'status': 'NOT_EVALUATED_RUNTIME_FAILURE',
            'error_type': type(exc).__name__, 'error': str(exc)}), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
