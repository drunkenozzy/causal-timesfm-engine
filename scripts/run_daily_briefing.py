"""Daily data -> raw TimesFM -> macro synthesis. Research context, no execution."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from scripts.daily_macro_synthesis import (context_hash, effective_cutoff, empty_briefing,
                                           synthesize, utc, write_briefing)
from scripts.shadow_data import ETH_SOURCE, TV_INSTRUMENTS, tv_source
from scripts.trust_features import available_at, load_histories, validate_histories


def neural_forecast(eth, checkpoint, horizon, mode):
    from scripts.timesfm_network_experiment import NeuralBaseline
    baseline = NeuralBaseline(checkpoint)
    context = eth.tail(512)
    if len(context) < 100 or not context.index.equals(pd.date_range(context.index[0], context.index[-1], freq='D')):
        raise ValueError('Neural ETH context must have at least 100 consecutive daily observations')
    _, full = baseline.model.forecast(inputs=[context.to_numpy()], freq=[0])
    quantiles = list(baseline.model.quantiles)
    result = {f'p{int(q * 100)}': float(full[0, horizon - 1, 1 + quantiles.index(q)]) for q in [0.1, 0.5, 0.9]}
    return {**result, 'asset': 'ETH-USD', 'quote': 'USD', 'origin_date': str(eth.index[-1].date()),
            'origin_close': float(eth.iloc[-1]), 'horizon_days': horizon,
            'target_close_utc': (eth.index[-1].tz_localize('UTC') + pd.Timedelta(days=horizon + 1)).isoformat(),
            'context_sha256': context_hash(context), 'quantile_kind': 'RAW_TIMESFM_TERMINAL',
            'produced_at_utc': datetime.now(timezone.utc).isoformat(), 'mode': mode,
            **baseline.provenance}


def load_inputs(directory):
    manifest = json.loads((Path(directory) / 'manifest.json').read_text(encoding='utf-8'))
    expected = {'ETH': ETH_SOURCE, **{name: tv_source(name) for name in TV_INSTRUMENTS}}
    if manifest.get('sources') != expected:
        raise ValueError('Source identity mismatch; no cross-asset or quote substitutions')
    if manifest.get('tv_session_rule') != 'recover UTC opening timestamp, then 17:00 America/New_York session roll':
        raise ValueError('Expected aligned session history collector contract')
    histories = load_histories(directory)
    validate_histories(histories)
    return histories, manifest


def run(args):
    mode = 'RETROSPECTIVE_REPLAY' if args.as_of else 'CURRENT_SNAPSHOT'
    as_of = utc(args.as_of) if args.as_of else pd.Timestamp.now(tz='UTC')
    report = empty_briefing(as_of)
    try:
        directory = args.history_dir
        if directory is None:
            from scripts.collect_trust_history import collect
            directory = Path(args.snapshot_root) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            collect(directory, args.tv_host_timezone)
            as_of = pd.Timestamp.now(tz='UTC')
        histories, manifest = load_inputs(directory)
        cutoff = effective_cutoff(as_of)
        eth = histories['ETH'].loc[available_at(histories['ETH'].index) <= cutoff]
        # First check data readiness before loading a large model.
        report = synthesize(histories, as_of, replay=bool(args.as_of))
        if report['status'] != 'DATA_NOT_READY':
            forecast = None
            forecast_error = None
            if args.checkpoint:
                try:
                    forecast = neural_forecast(eth, args.checkpoint, args.horizon_days, mode)
                except Exception as exc:
                    forecast_error = f'{type(exc).__name__}: {exc}'
            if not args.as_of:
                as_of = pd.Timestamp.now(tz='UTC')
            report = synthesize(histories, as_of, forecast, replay=bool(args.as_of))
            if forecast_error:
                report['forecast'] = {'status': 'UNAVAILABLE', 'reason': forecast_error}
        report['input_manifest'] = manifest
        report['history_directory'] = str(directory)
    except (Exception, SystemExit) as exc:
        report = empty_briefing(as_of, f'{type(exc).__name__}: {exc}')
    report['mode'] = mode
    report['generated_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['refresh_due_utc'] = (effective_cutoff(as_of) + pd.Timedelta(days=1)).isoformat() if not args.as_of else None
    write_briefing(args.output, report)
    print(json.dumps({'output': str(args.output), 'status': report['status'], 'regime': report['market_regime']['label']}))
    return 2 if report['status'] == 'DATA_NOT_READY' else 1 if report['status'] == 'PARTIAL' else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-dir', type=Path, help='Reuse a verified aligned snapshot instead of collecting')
    parser.add_argument('--snapshot-root', default='data/macro_snapshots')
    parser.add_argument('--tv-host-timezone', help='Required for collection; IANA timezone of the tvdatafeed host')
    parser.add_argument('--checkpoint', type=Path, help='Local TimesFM 2.0 500M PyTorch checkpoint; omission yields PARTIAL')
    parser.add_argument('--horizon-days', type=int, choices=range(1, 91), default=1)
    parser.add_argument('--as-of', help='Explicit timezone required; marks output RETROSPECTIVE_REPLAY')
    parser.add_argument('--output', type=Path, default=Path('data/MACRO_SYNTHESIS.json'))
    args = parser.parse_args()
    if not args.history_dir and (not args.tv_host_timezone or args.as_of):
        parser.error('Collection requires --tv-host-timezone; replay requires --history-dir')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
