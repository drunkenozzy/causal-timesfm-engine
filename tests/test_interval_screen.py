import math
import numpy as np
import pandas as pd
import pytest
from scripts.interval_screen import CONFIG,build_records,score_records,paired_interval,load_snapshot


class NeuralFixture:
    def forecast(self,history,horizon_days):
        assert horizon_days==7
        return dict(timesfm_executed=True,degraded_mode=False,
            p10_downside=history[-1]*.94,p50_expected=history[-1],p90_upside=history[-1]*1.06)


def sample():
    dates=pd.date_range('2025-01-01',periods=160).strftime('%Y-%m-%d').tolist()
    prices=np.array([100*math.exp(.01*math.sin(i/4)+.001*i) for i in range(160)])
    return dates,prices


def test_only_matured_h7_outcomes_calibrate_ordered_intervals():
    dates,prices=sample()
    rows=build_records(dates,prices,NeuralFixture())
    for r in rows:
        assert r['latest_training_target'] is None or r['latest_training_target']<=r['origin_date']
        assert r['target_index']-r['origin_index']==7
        assert r['forecasts']['residual_interval'][1]==r['forecasts']['timesfm'][1]
        assert 0<r['forecasts']['residual_interval'][0]<=r['forecasts']['residual_interval'][1]<=r['forecasts']['residual_interval'][2]
    changed=prices.copy(); changed[120:]*=2
    rerun=build_records(dates,changed,NeuralFixture())
    assert [r['forecasts'] for r in rows if r['origin_index']<120]==[r['forecasts'] for r in rerun if r['origin_index']<120]
    result=score_records(rows,dict(CONFIG,bootstrap_repetitions=20))
    assert result['evaluation_origins']==sum(r['eligible'] for r in rows)
    assert not result['promotion_allowed']
    assert result['comparisons']['timesfm']['block_length']>=7


def test_block_bootstrap_reports_uncertainty_and_short_samples():
    assert paired_interval([1]*10,[.5]*10)['status']=='INSUFFICIENT_BLOCKS'
    assert paired_interval([1]*50,[.5]*50,reps=20)['ci95']==[.5,.5]


def test_snapshot_rejects_gaps_duplicates_and_invalid_prices(tmp_path):
    for dates,prices in [(['2025-01-01','2025-01-03'],[1,2]),(['2025-01-01']*2,[1,2]),(['2025-01-01','2025-01-02'],[1,-1])]:
        p=tmp_path/'input.csv'
        pd.DataFrame(dict(Date=dates,Close=prices)).to_csv(p,index=False)
        with pytest.raises(ValueError):load_snapshot(p)
