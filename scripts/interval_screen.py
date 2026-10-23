"""Offline h=7 exploratory interval screen. No downloads, orders or hosted jobs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.calibration import compute_pinball_loss, compute_winkler_interval_score
from core.engine_timesfm import TimesFmBaselineEngine

CONFIG=dict(experiment='BTC_H7_INTERVAL_EXPLORATORY_001',horizon=7,minimum_context=60,
    calibration_minimum=20,calibration_window=120,interval_shrinkage=.5,
    bootstrap_block=14,bootstrap_repetitions=1000,seed=42,
    primary_metric='mean_spot_normalized_80pct_interval_score',
    arms=['timesfm','volatility','residual_interval'],evaluation_type='RETROSPECTIVE_EXPLORATORY')


def load_snapshot(path):
    df=pd.read_csv(path)
    if list(df.columns)!=['Date','Close']:
        raise ValueError('Expected Date,Close snapshot')
    dates=pd.to_datetime(df.Date,errors='raise',utc=True)
    if dates.duplicated().any() or not dates.is_monotonic_increasing or not dates.diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError('Snapshot must contain unique consecutive daily closes')
    prices=df.Close.to_numpy(dtype=float)
    if not np.isfinite(prices).all() or np.any(prices<=0):
        raise ValueError('Invalid snapshot prices')
    return df.Date.tolist(),prices


def build_records(dates,prices,engine,config=None):
    cfg=dict(CONFIG if config is None else config)
    h=cfg['horizon']
    records=[]
    for origin in range(cfg['minimum_context']-1,len(prices)-h):
        prefix=np.asarray(prices[:origin+1],dtype=float)
        forecast=engine.forecast(prefix.tolist(),horizon_days=h)
        if not forecast.get('timesfm_executed') or forecast.get('degraded_mode',True):
            raise RuntimeError('Neural inference failed; no fallback or dropped origins')
        lower,median,upper=[float(forecast[k]) for k in ['p10_downside','p50_expected','p90_upside']]
        if not all(math.isfinite(v) and v>0 for v in (lower,median,upper)) or not lower<=median<=upper:
            raise ValueError('Invalid neural quantiles')
        spot=float(prefix[-1])
        sigma=float(np.std(np.diff(np.log(prefix[-61:])),ddof=1))
        radius=1.2815515655446004*sigma*math.sqrt(h)
        vol=[spot*math.exp(-radius),spot,spot*math.exp(radius)]
        # Outcomes that close today may train a forecast issued after today's close.
        known=[r for r in records if r['target_index']<=origin][-cfg['calibration_window']:]
        ready=len(known)>=cfg['calibration_minimum']
        adjusted=[lower,median,upper]
        if ready:
            residual=np.array([math.log(r['actual']/r['forecasts']['timesfm'][1]) for r in known])
            qlo,qhi=np.quantile(residual,[.1,.9])
            w=cfg['interval_shrinkage']
            # Positive log-widths preserve the median and ordered positive bounds.
            down=(1-w)*math.log(median/lower)+w*max(0.,-float(qlo))
            up=(1-w)*math.log(upper/median)+w*max(0.,float(qhi))
            adjusted=[median*math.exp(-down),median,median*math.exp(up)]
        records.append(dict(origin_index=origin,target_index=origin+h,origin_date=dates[origin],
            target_date=dates[origin+h],origin_close=spot,actual=float(prices[origin+h]),
            history_sha256=hashlib.sha256(prefix.astype('<f8').tobytes()).hexdigest(),
            training_count=len(known),latest_training_target=max((r['target_date'] for r in known),default=None),
            eligible=ready,forecasts=dict(timesfm=[lower,median,upper],volatility=vol,residual_interval=adjusted)))
    return records


def paired_interval(reference,candidate,block=14,reps=1000,seed=42):
    """Paired contiguous-block resampling of loss DIFFERENCES, no iid shortcut."""
    delta=np.asarray(reference)-np.asarray(candidate)
    if len(delta)<2*block:
        return dict(status='INSUFFICIENT_BLOCKS',block_length=block)
    rng=np.random.default_rng(seed)
    means=[]
    for _ in range(reps):
        starts=rng.integers(0,len(delta)-block+1,size=math.ceil(len(delta)/block))
        ids=np.concatenate([np.arange(s,s+block) for s in starts])[:len(delta)]
        means.append(float(delta[ids].mean()))
    return dict(status='DESCRIPTIVE_ONLY',mean_reference_minus_candidate=float(delta.mean()),
        ci95=np.quantile(means,[.025,.975]).tolist(),block_length=block,replications=reps)


def score_records(records,config=None):
    cfg=dict(CONFIG if config is None else config)
    selected=[r for r in records if r['eligible']]
    if not selected:
        raise ValueError('Insufficient matured outcomes for common evaluation sample')
    losses={arm:[] for arm in cfg['arms']}
    metrics={}
    for arm in cfg['arms']:
        below=above=0
        pinlo=[]; pinhi=[]; widths=[]; mederr=[]
        for row in selected:
            lo,med,hi=row['forecasts'][arm]; actual=row['actual']; spot=row['origin_close']
            if not 0<lo<=med<=hi or not all(math.isfinite(v) for v in [lo,med,hi,actual,spot]):
                raise ValueError('Invalid interval record')
            # Normalize BEFORE scoring to avoid pre-normalization rounding differences.
            losses[arm].append(compute_winkler_interval_score(actual/spot,lo/spot,hi/spot))
            pinlo.append(compute_pinball_loss(actual/spot,lo/spot,.1))
            pinhi.append(compute_pinball_loss(actual/spot,hi/spot,.9))
            widths.append((hi-lo)/spot); mederr.append(abs(actual-med)/spot)
            below+=actual<lo; above+=actual>hi
        n=len(selected)
        metrics[arm]=dict(mean_interval_score=float(np.mean(losses[arm])),
            mean_pinball_p10=float(np.mean(pinlo)),mean_pinball_p90=float(np.mean(pinhi)),
            coverage=(n-below-above)/n,below_p10_rate=below/n,above_p90_rate=above/n,
            mean_relative_width=float(np.mean(widths)),median_relative_mae=float(np.mean(mederr)))
    comparisons={reference:paired_interval(losses[reference],losses['residual_interval'],
        block=cfg['bootstrap_block'],reps=cfg['bootstrap_repetitions'],seed=cfg['seed'])
        for reference in ['timesfm','volatility']}
    return dict(evaluation_origins=len(selected),first_origin=selected[0]['origin_date'],
        last_origin=selected[-1]['origin_date'],overlapping_targets=True,
        nonoverlapping_origin_count=len(selected[::cfg['horizon']]),metrics=metrics,comparisons=comparisons,
        promotion_allowed=False,verdict='EXPLORATORY_ONLY_NO_CONFIRMATORY_CLAIM')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,default=ROOT/'data/btc_protocol_002_snapshot.csv')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Use a new output path; existing experiments are immutable')
    if not args.checkpoint.is_file(): raise FileNotFoundError('Local checkpoint required; no downloads')
    with args.checkpoint.open('rb') as handle:
        digest=hashlib.file_digest(handle,'sha256').hexdigest()
    if digest!=args.checkpoint_sha256: raise ValueError('Checkpoint hash mismatch')
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    torch.set_num_threads(4); torch.manual_seed(CONFIG['seed'])
    dates,prices=load_snapshot(args.snapshot)
    engine=TimesFmBaselineEngine(checkpoint_path=args.checkpoint)
    records=build_records(dates,prices,engine)
    result=dict(config=CONFIG,generated_at_utc=datetime.now(timezone.utc).isoformat(),
        input_sha256=hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),checkpoint_sha256=digest,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'core/engine_timesfm.py',ROOT/'core/calibration.py']},
        provenance='Frozen retrospective prices; historical availability/vintages not independently established. No macro effect tested.',
        summary=score_records(records),records=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__': main()
