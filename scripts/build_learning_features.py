"""Build registered Trust inputs from local, timestamped series. No network access."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.adaptive_research import utc


def build(series, decision_utc):
    decision=pd.Timestamp(utc(decision_utc))
    known={}
    for name, rows in series.items():
        frame=pd.DataFrame(rows)
        frame['observed_at_utc']=pd.to_datetime(frame.observed_at_utc,utc=True)
        frame['available_at_utc']=pd.to_datetime(frame.available_at_utc,utc=True)
        if (frame.available_at_utc < frame.observed_at_utc).any(): raise ValueError('Invalid publication ordering')
        frame=frame.loc[(frame.available_at_utc<=decision)&(frame.observed_at_utc<=decision)].sort_values('observed_at_utc')
        if frame.empty or decision-frame.observed_at_utc.iloc[-1]>pd.Timedelta(days=4): continue
        if frame.observed_at_utc.duplicated().any() or not np.isfinite(frame.value).all(): raise ValueError('Invalid series')
        if name!='US10Y' and (frame.value<=0).any(): raise ValueError('Nonpositive market level')
        if frame.source.isna().any() or not frame.source.astype(str).str.len().gt(0).all(): raise ValueError('Missing source')
        known[name]=frame
    result={}
    def emit(key,value,names):
        frames=[known[n] for n in names]
        # Oldest contributing latest observation makes stale mixed inputs visible.
        result[key]=dict(value=float(value),observed_at_utc=min(f.observed_at_utc.iloc[-1] for f in frames).isoformat(),
                         available_at_utc=max(f.available_at_utc.max() for f in frames).isoformat(),
                         source='local-series-sha256:'+hashlib.sha256(''.join(f.to_json(date_format='iso') for f in frames).encode()).hexdigest())
    for name,key in [('DXY','dxy_return_5d'),('target','target_return_5d'),('US10Y','us10y_change_5d_pp')]:
        if name in known and len(known[name])>=6:
            values=known[name].value
            value=values.iloc[-1]-values.iloc[-6] if name=='US10Y' else np.log(values.iloc[-1]/values.iloc[-6])
            emit(key,value,[name])
    if 'DXY' in known and len(known['DXY'])>=2:
        values=known['DXY'].value
        emit('dxy_return_1d',np.log(values.iloc[-1]/values.iloc[-2]),['DXY'])
    if 'Brent' in known and 'WTI' in known:
        # Pair actual observed dates; never forward-fill a missing market close.
        b=known['Brent'].set_index('observed_at_utc').value
        w=known['WTI'].set_index('observed_at_utc').value
        pair=pd.concat([b,w],axis=1,join='inner').dropna()
        if not pair.empty and decision-pair.index[-1]<=pd.Timedelta(days=4):
            emit('brent_wti_spread_pct',100*(pair.iloc[-1,0]/pair.iloc[-1,1]-1),['Brent','WTI'])
            result['brent_wti_spread_pct']['observed_at_utc']=pair.index[-1].isoformat()
    required=['target','DXY','Brent','USDTRY']
    if all(n in known for n in required):
        panel=pd.concat({n:known[n].set_index('observed_at_utc').value for n in required},axis=1,join='inner').dropna()
        if len(panel)>=61 and decision-panel.index[-1]<=pd.Timedelta(days=4):
            # Identical timestamps required: asynchronous closes need a separately reviewed alignment.
            returns=np.log(panel).diff().dropna().tail(60)
            correlation=returns.target.corr(returns.DXY)
            if np.isfinite(correlation):
                emit('signed_dollar_shock',correlation*returns.DXY.iloc[-1],required)
            risk=returns[['target','Brent','USDTRY']].copy();risk['USDTRY']*=-1
            down=risk<0
            excess=[]
            for i in range(3):
                for j in range(i):
                    a,b=down.iloc[:,i],down.iloc[:,j]
                    excess.append((a&b).mean()-a.mean()*b.mean())
            emit('downside_connectivity',np.mean(excess),required)
            for key in ['signed_dollar_shock','downside_connectivity']:
                if key in result: result[key]['observed_at_utc']=panel.index[-1].isoformat()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--decision-utc',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=build(json.loads(args.input.read_text()),args.decision_utc)
    with args.output.open('x',encoding='utf-8') as handle:json.dump(result,handle,indent=2,allow_nan=False)
    print('Built '+', '.join(result)+'; unavailable inputs omitted, never filled with zero.')


if __name__=='__main__':main()
