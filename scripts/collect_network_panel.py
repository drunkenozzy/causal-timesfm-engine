"""Snapshot the same six non-BTC provider series used by the shadow experiments."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from scripts.shadow_data import ETH_SOURCE, TV_INSTRUMENTS, fetch_eth_close, fetch_tv_close, tv_source, snapshot_hash
from scripts.network_experiment import validate_panel


def collect():
    started = datetime.now(timezone.utc)
    sources = {'ETH': ETH_SOURCE, **{name: tv_source(name) for name in TV_INSTRUMENTS}}
    frames = {'ETH': fetch_eth_close()}
    frames.update({name: fetch_tv_close(*instrument) for name, instrument in TV_INSTRUMENTS.items()})
    frames = {name: frame.loc[frame.index < pd.Timestamp(started.date())] for name, frame in frames.items()}
    missing = [name for name, frame in frames.items() if frame.empty]
    manifest = {'experiment': 'MARKET_NETWORK_SCREEN_001', 'collected_at_utc': started.isoformat(),
                'sources': sources, 'missing': missing, 'status': 'DATA_NOT_READY' if missing else 'READY',
                'alignment': 'complete common session dates; no forward/back fill',
                'availability': 'retrospective provider snapshot, not point-in-time vintage evidence',
                'series': {name: {'rows': len(frame), 'sha256': snapshot_hash(frame, sources[name])}
                           for name, frame in frames.items()}}
    if missing:
        return None, manifest
    panel = pd.concat([frame.Close.rename(name) for name, frame in frames.items()], axis=1, join='inner').sort_index()
    panel = validate_panel(panel)
    manifest.update(rows=len(panel), input_sha256=snapshot_hash(panel, sources))
    if len(panel) < 151:
        manifest['status'] = 'INSUFFICIENT_HISTORY'
    return panel, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, help='New directory for this snapshot (must not exist)')
    args = parser.parse_args()
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    panel, manifest = collect()
    if panel is not None:
        panel.to_csv(destination / 'panel.csv', index_label='Date')
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
    if manifest['status'] != 'READY':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
