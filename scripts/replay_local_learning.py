"""Exercise the learning journal on saved h=7 neural forecasts, never prospective evidence."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.adaptive_research import LocalLearningLedger, utc, digest, RECIPE


def replay(report,journal_path):
    if Path(journal_path).exists(): raise FileExistsError('Replay needs a new journal')
    ledger=LocalLearningLedger(journal_path,mode='RETROSPECTIVE_REPLAY')
    pending=[]
    scored=[]
    for row in report['records']:
        decision=datetime.fromisoformat(row['origin_date']).replace(tzinfo=timezone.utc)+timedelta(days=1)
        for identifier,terminal,actual in pending:
            if terminal<=decision and identifier not in ledger.resolutions:
                ledger.resolve(identifier,actual,'saved-h7-study',now=decision.isoformat())
        terminal=datetime.fromisoformat(row['target_date']).replace(tzinfo=timezone.utc)+timedelta(days=1)
        lo,med,hi=row['forecasts']['timesfm']
        packet=dict(asset='BTC-USD',horizon_days=7,decision_utc=decision.isoformat(),terminal_utc=terminal.isoformat(),
                    origin_utc=decision.isoformat(),
                    origin_close=row['origin_close'],p10=lo,p50=med,p90=hi,features={},
                    timesfm_executed=True,degraded_mode=False,source={'saved_input_sha256':digest(report)})
        record=ledger.register(packet,now=decision.isoformat())
        pending.append((record['id'],terminal,row['actual']))
        scored.append(dict(actual=row['actual'],spot=row['origin_close'],timesfm=med,
                           persistence=row['origin_close'],adaptive=record['result']['prediction'],
                           selected=record['result']['selected'],trained=bool(record['result']['training_counts'])))
    for identifier,terminal,actual in pending:
        if identifier not in ledger.resolutions:
            ledger.resolve(identifier,actual,'saved-h7-study',now=max(t for _,t,_ in pending).isoformat())
    eligible=[r for r in scored if r['trained']]
    scores={arm:float(np.mean([((r['actual']-r[arm])/r['spot'])**2 for r in eligible]))
            for arm in ['timesfm','persistence','adaptive']} if eligible else {}
    return dict(mode='RETROSPECTIVE_REPLAY',recipe_sha256=RECIPE,input_content_sha256=digest(report),
                total_origins=len(scored),post_training_origins=len(eligible),
                normalized_mspe=scores,selected_counts={a:sum(r['selected']==a for r in scored) for a in ['timesfm','persistence','bias','raw','network']},
                macro_features_tested=False,promotion_allowed=False,
                note='Overlapping seven-day targets; functional replay, not independent evidence or a significance test.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'docs/BTC_H7_INTERVAL_EXPLORATORY_001.json')
    parser.add_argument('--journal',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a new report path')
    result=replay(json.loads(args.input.read_text()),args.journal)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
