"""Isolated five-arm TimesFM Trust proxy screen, never imported by daily pipelines."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from scripts.network_experiment import ridge_prediction
from scripts.timesfm_network_experiment import NeuralBaseline, paired_skill_interval, sha256_file
from scripts.trust_features import CONFIG, NAMES, SIGNED, TARGETS, TRUST, available_at, features_at, load_histories, validate_histories


def plan_target(histories, target):
    validate_histories(histories)
    if target not in TARGETS:
        raise ValueError('Unregistered target')
    eligible, excluded = [], []
    # Two pre-evaluation years are ample for the fixed 252 completed-residual warm-up.
    planning_start = pd.Timestamp(CONFIG['evaluation_start']) - pd.DateOffset(years=2)
    for date in histories[target].index[histories[target].index >= planning_start]:
        try:
            values, audit, context = features_at(histories, target, date)
            if target == 'ETH' and (date - context.index[-1]).days != 1:
                raise ValueError('MISSING_DAILY_TARGET_BAR')
            context = context.tail(CONFIG['context_observations'])
            if len(context) < 100:
                raise ValueError('SHORT_NEURAL_CONTEXT')
            eligible.append({'target_date': date.strftime('%Y-%m-%d'), 'features': values, 'audit': audit,
                             'context': context.to_numpy(dtype=float).tolist(),
                             'current_price': float(context.iloc[-1]), 'actual_price': float(histories[target].loc[date])})
        except ValueError as exc:
            excluded.append({'target_date': date.strftime('%Y-%m-%d'), 'reason': str(exc)})
    pre = [r for r in eligible if r['target_date'] < CONFIG['evaluation_start']]
    evaluation = [r for r in eligible if r['target_date'] >= CONFIG['evaluation_start']]
    if len(pre) < CONFIG['min_residual_train']:
        raise ValueError(f'{target}: insufficient pre-evaluation residual history ({len(pre)})')
    selected = pre[-CONFIG['min_residual_train']:] + evaluation
    diagnostics = {'target': target, 'warmup_origins': CONFIG['min_residual_train'],
                   'evaluation_origins': len(evaluation), 'first_evaluation': evaluation[0]['target_date'] if evaluation else None,
                   'last_evaluation': evaluation[-1]['target_date'] if evaluation else None,
                   'excluded_counts': dict(Counter(row['reason'] for row in excluded)), 'excluded': excluded}
    return selected, diagnostics


def evaluate_records(records, config=None):
    settings = dict(CONFIG if config is None else config)
    if not records:
        raise ValueError('No forecast records')
    if settings['min_residual_train'] < 20 or settings['residual_train_window'] < settings['min_residual_train'] or settings['ridge_penalty'] <= 0:
        raise ValueError('Invalid residual training configuration')
    if any(records[i]['target_date'] >= records[i + 1]['target_date'] for i in range(len(records) - 1)):
        raise ValueError('Forecast records must have unique ordered target dates')
    for row in records:
        expected = pd.Timestamp(row['target_date']).tz_localize('UTC') + pd.Timedelta(hours=CONFIG['availability_hour_utc'])
        if pd.Timestamp(row['audit']['decision_utc']) != expected:
            raise ValueError('Decision time must match the registered target-session cutoff')
    features = pd.DataFrame([r['features'] for r in records])
    if set(SIGNED + TRUST) - set(features):
        raise ValueError('Missing registered Trust features')
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError('Nonfinite features')
    raw = [c for c in features if c not in SIGNED + TRUST]
    groups = {'raw_controls': raw, 'raw_signed': raw + SIGNED, 'raw_trust': raw + TRUST,
              'raw_signed_trust': raw + SIGNED + TRUST}
    neural = np.asarray([r['neural_prediction'] for r in records], dtype=float)
    current = np.asarray([r['current_price'] for r in records], dtype=float)
    actual_price = np.asarray([r['actual_price'] for r in records], dtype=float)
    if not np.isfinite(np.concatenate([neural, current, actual_price])).all() or min(neural.min(), current.min(), actual_price.min()) <= 0:
        raise ValueError('Invalid prices or neural forecasts')
    base_returns = np.log(neural / current)
    actual = np.log(actual_price / current)
    residuals = actual - base_returns
    results = []
    for i, row in enumerate(records):
        if row['target_date'] < settings['evaluation_start']:
            continue
        cutoff = pd.Timestamp(row['audit']['decision_utc'])
        # Availability, rather than row position alone, decides which outcomes may train the correction.
        train = [j for j in range(i) if available_at([records[j]['target_date']])[0] <= cutoff]
        train = train[-settings['residual_train_window']:]
        if len(train) < settings['min_residual_train']:
            raise ValueError('Incomplete residual training at a registered evaluation origin')
        forecasts = {'timesfm': float(base_returns[i])}
        for arm, columns in groups.items():
            correction = ridge_prediction(features[columns].iloc[train].to_numpy(), residuals[train],
                                          features[columns].iloc[i].to_numpy(), settings['ridge_penalty'])
            forecasts[arm] = float(base_returns[i] + correction)
        if not np.isfinite(list(forecasts.values())).all():
            raise ValueError('Invalid corrected forecasts')
        results.append({'target_date': row['target_date'], 'decision_utc': row['audit']['decision_utc'],
                        'actual_log_return': float(actual[i]), 'actual_price': float(actual_price[i]),
                        'current_price': float(current[i]), 'forecasts_log_return': forecasts,
                        'training_count': len(train), 'last_training_target': records[train[-1]]['target_date']})
    losses = {a: [(r['actual_log_return'] - r['forecasts_log_return'][a]) ** 2 for r in results] for a in settings['arms']}
    scores = {a: float(np.mean(v)) for a, v in losses.items()} if results else {}
    price_scores = {a: float(np.mean([(r['actual_price'] - r['current_price'] * np.exp(r['forecasts_log_return'][a])) ** 2 for r in results]))
                    for a in settings['arms']} if results else {}
    comparisons = {}
    for candidate in groups:
        for reference in ['timesfm', 'raw_controls']:
            if candidate == reference or not results:
                continue
            denominator = scores[reference]
            comparisons[f'{candidate}_vs_{reference}'] = {
                'skill_pct': 100 * (1 - scores[candidate] / denominator) if denominator > 0 else None,
                'descriptive_95pct_interval': paired_skill_interval(losses[reference], losses[candidate],
                    settings['bootstrap_replications'], settings['bootstrap_block_length'], settings['seed'])}
    yearly = {}
    for year in sorted({r['target_date'][:4] for r in results}):
        subset = [r for r in results if r['target_date'].startswith(year)]
        yearly[year] = {'origins': len(subset), 'mspe': {a: float(np.mean([(r['actual_log_return'] - r['forecasts_log_return'][a]) ** 2 for r in subset]))
                                                       for a in settings['arms']}}
    return {'status': 'EXPLORATORY' if len(results) >= 252 else 'INSUFFICIENT_EVALUATION_ORIGINS',
            'evaluation_origins': len(results), 'raw_features': raw, 'signed_features': SIGNED, 'trust_features': TRUST,
            'mspe': scores, 'price_mspe': price_scores, 'comparisons': comparisons, 'calendar_year_results': yearly,
            'promotion': 'NOT_PROMOTED_RESEARCH_ONLY', 'predictions': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--histories', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if not args.plan_only and not args.checkpoint:
        parser.error('--checkpoint is required for neural execution')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    histories = load_histories(args.histories)
    manifest = {'config': CONFIG, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'input_manifest': json.loads((Path(args.histories) / 'manifest.json').read_text()),
                'source_hashes': {n: sha256_file(Path(__file__).with_name(n)) for n in
                                  ['trust_screen.py', 'trust_features.py', 'collect_trust_history.py', 'shadow_data.py',
                                   'timesfm_network_experiment.py', 'network_experiment.py']},
                'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                'limitations': ['Historical availability is a declared conservative assumption, not provider publication evidence.',
                                'Current provider vintage may contain revisions; past data are not an untouched holdout.',
                                'Risk/safety price proxies do not identify investor trust or observed capital flows.',
                                'Multiple asset/feature comparisons are exploratory; intervals are descriptive.']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    try:
        plans = {}
        for target in TARGETS:
            rows, diagnostics = plan_target(histories, target)
            plans[target] = rows
            (output / f'coverage_{target}.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
            print(json.dumps({k: v for k, v in diagnostics.items() if k != 'excluded'}), flush=True)
        if args.plan_only:
            return
        model = NeuralBaseline(args.checkpoint)
        smoke = model.predict(np.linspace(100., 110., 100))
        (output / 'runtime.json').write_text(json.dumps({'runtime': model.provenance, 'synthetic_smoke': smoke}, indent=2), encoding='utf-8')
        for target, plan in plans.items():
            records = []
            with (output / f'neural_{target}.jsonl').open('x', encoding='utf-8') as handle:
                for i, row in enumerate(plan):
                    context = np.asarray(row['context'], dtype=float)
                    record = {k: v for k, v in row.items() if k != 'context'}
                    record.update(neural_prediction=model.predict(context), context_rows=len(context),
                                  context_sha256=hashlib.sha256(context.tobytes()).hexdigest())
                    records.append(record)
                    handle.write(json.dumps(record, allow_nan=False) + '\n')
                    handle.flush()
                    if (i + 1) % 50 == 0:
                        print(f'{target}: {i + 1}/{len(plan)} neural forecasts; elapsed {time.monotonic() - started:.0f}s', flush=True)
            result = evaluate_records(records)
            result['timesfm_executed'], result['degraded_mode'] = True, False
            (output / f'result_{target}.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
            print(json.dumps({k: v for k, v in result.items() if k not in ['predictions', 'raw_features']}), flush=True)
        (output / 'COMPLETED.json').write_text(json.dumps({'elapsed_seconds': time.monotonic() - started}), encoding='utf-8')
    except Exception as exc:
        (output / 'FAILED.json').write_text(json.dumps({'status': 'NOT_EVALUATED', 'error': f'{type(exc).__name__}: {exc}'}), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
