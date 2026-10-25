"""Run one offline daily crypto forecast and update its local correction journal."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.adaptive_research import LocalLearningLedger, utc
from scripts.interval_screen import load_snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history',type=Path,required=True,help='Local Date,Close CSV of consecutive completed UTC days')
    parser.add_argument('--asset',choices=['BTC-USD','ETH-USD'],required=True)
    parser.add_argument('--horizon',type=int,choices=[1,7],default=7)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--ledger',type=Path,required=True)
    parser.add_argument('--features',type=Path,help='Optional timestamped feature packets; see documentation')
    args=parser.parse_args()
    dates,prices=load_snapshot(args.history)
    start=datetime.now(timezone.utc)
    if len(prices)<60: raise ValueError('At least 60 daily closes required')
    if dates[-1]!=(start.date()-timedelta(days=1)).isoformat():
        raise ValueError('History must end on yesterday UTC; partial/stale closes are refused')
    if not args.checkpoint.is_file(): raise FileNotFoundError('Local checkpoint required; no downloads')
    with args.checkpoint.open('rb') as handle:
        checkpoint_hash=hashlib.file_digest(handle,'sha256').hexdigest()
    if checkpoint_hash!=args.checkpoint_sha256: raise ValueError('Checkpoint hash mismatch')
    history_hash=hashlib.sha256(args.history.read_bytes()).hexdigest()
    features=json.loads(args.features.read_text()) if args.features else {}
    journal=LocalLearningLedger(args.ledger)
    by_date=dict(zip(dates,prices))
    for identifier,event in list(journal.predictions.items()):
        if event['packet']['asset']!=args.asset or identifier in journal.resolutions: continue
        terminal=utc(event['packet']['terminal_utc'])
        target=(terminal-timedelta(days=1)).date().isoformat()
        if terminal<=start and target in by_date:
            journal.resolve(identifier,float(by_date[target]),'local-close-snapshot:'+history_hash)
    terminal=datetime.combine(start.date()+timedelta(days=args.horizon),datetime.min.time(),tzinfo=timezone.utc)
    previous=[p for p in journal.predictions.values() if p['packet']['asset']==args.asset
              and p['packet']['horizon_days']==args.horizon and utc(p['packet']['terminal_utc'])==terminal]
    if previous:
        print(json.dumps({'status':'ALREADY_REGISTERED','forecast':previous[0]},indent=2));return
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from core.engine_timesfm import TimesFmBaselineEngine
    torch.set_num_threads(4)
    model=TimesFmBaselineEngine(checkpoint_path=args.checkpoint)
    forecast=model.forecast(prices.tolist(),horizon_days=args.horizon)
    now=datetime.now(timezone.utc)
    if now.date()!=start.date(): raise ValueError('UTC date rolled over; rerun with current history')
    packet=dict(asset=args.asset,horizon_days=args.horizon,decision_utc=now.isoformat(),
                origin_utc=datetime.combine(start.date(),datetime.min.time(),tzinfo=timezone.utc).isoformat(),
                terminal_utc=terminal.isoformat(),origin_close=float(prices[-1]),
                p10=forecast['p10_downside'],p50=forecast['p50_expected'],p90=forecast['p90_upside'],
                timesfm_executed=forecast['timesfm_executed'],degraded_mode=forecast['degraded_mode'],
                source={'checkpoint_sha256':checkpoint_hash,'history_sha256':history_hash},features=features)
    print(json.dumps(journal.register(packet),indent=2,allow_nan=False))


if __name__=='__main__':main()
