from datetime import datetime, timedelta, timezone
import math
import pytest
from scripts.build_learning_features import build


def series():
    result={}
    for j,name in enumerate(['target','DXY','Brent','WTI','USDTRY','US10Y']):
        result[name]=[]
        for i in range(70):
            date=(datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(days=i)).isoformat()
            result[name].append(dict(observed_at_utc=date,available_at_utc=date,source='test',
                                    value=(4+.01*i if name=='US10Y' else 100*math.exp(.01*math.sin(i+j)))))
    return result


def test_future_revisions_cannot_enter_features():
    s=series();decision=s['target'][-3]['observed_at_utc'];before=build(s,decision)
    for rows in s.values():
        rows[-1]['value']*=1000
        rows[-2]['value']*=1000
    assert build(s,decision)==before
    assert len(before)==7
    assert before['us10y_change_5d_pp']['value']==pytest.approx(.05)


def test_missing_or_asynchronous_market_data_disables_network():
    s=series();decision=s['target'][-1]['observed_at_utc']
    del s['USDTRY']
    result=build(s,decision)
    assert 'dxy_return_5d' in result and 'downside_connectivity' not in result
    s=series()
    for row in s['DXY']:
        row['observed_at_utc']=row['observed_at_utc'].replace('00:00:00','01:00:00')
        row['available_at_utc']=row['observed_at_utc']
    assert 'signed_dollar_shock' not in build(s,decision)


def test_stale_series_is_not_fabricated_as_neutral():
    s=series();decision=s['target'][-1]['observed_at_utc'];s['DXY']=s['DXY'][:-10]
    result=build(s,decision)
    assert 'dxy_return_5d' not in result
    assert 'signed_dollar_shock' not in result
