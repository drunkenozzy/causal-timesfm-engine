import math
import pytest
from core.calibration import run_baseline_tournament, compute_winkler_interval_score
from core.markov_regime import InstitutionalMarkovEngine
from core.onchain_liquidity import OnChainLiquidityEngine


class FakeNeural:
    def __init__(self, fail_at=None):
        self.calls=[]
        self.fail_at=fail_at

    def forecast(self, history, horizon_days):
        self.calls.append((list(history),horizon_days))
        ok=len(self.calls)!=self.fail_at
        return dict(p50_expected=history[-1]*1.1,engine='test-neural',timesfm_executed=ok,degraded_mode=not ok)


def test_tournament_scores_each_actual_neural_call_and_horizon():
    values=[100+i for i in range(55)]
    model=FakeNeural()
    result=run_baseline_tournament(values,h=7,min_train_len=10,forecast_engine=model)
    assert len(model.calls)==len(result['origins'])==39
    for record,(history,horizon) in zip(result['origins'],model.calls):
        assert history==values[:record['origin_index']+1]
        assert horizon==7 and record['target_index']==record['origin_index']+7
        assert record['m1']==history[-1]*1.1
        if record['latest_training_target'] is not None:
            assert record['latest_training_target']<=record['origin_index']
    expected=sum((r['actual']-r['m1'])**2 for r in result['origins'])/39
    assert result['models']['TimesFM-Neural']['mspe']==pytest.approx(expected)


def test_later_neural_failure_invalidates_entire_comparison():
    result=run_baseline_tournament(list(range(100,150)),min_train_len=10,forecast_engine=FakeNeural(fail_at=3))
    assert result['status']=='NOT_EVALUATED_RUNTIME_FAILURE'
    assert not result['TIMESFM_COMPARISON_VALID'] and 'models' not in result


def test_future_actuals_cannot_change_earlier_selected_weights():
    values=[100*math.exp(.001*i+.02*math.sin(i)) for i in range(130)]
    base=run_baseline_tournament(values,h=7,forecast_engine=FakeNeural())['origins']
    changed=values[:100]+[v*3 for v in values[100:]]
    after=run_baseline_tournament(changed,h=7,forecast_engine=FakeNeural())['origins']
    for a,b in zip(base,after):
        if a['origin_index']<100:
            assert a['m4']==b['m4'] and a['alpha']==b['alpha']


def test_staleness_and_pre_history_requests_fail_closed(tmp_path):
    model=OnChainLiquidityEngine(cache_file=str(tmp_path/'absent.json'))
    model.history_by_date={'2026-09-17':100.}
    stale=model.get_liquidity_features_as_of('2026-10-20')
    assert stale['data_status']=='DATA_STALE_CRITICAL' and stale['staleness_days']==33
    assert stale['z_score'] is None and not stale['decoupling_active']
    before=model.get_liquidity_features_as_of('2020-01-01')
    assert before['as_of_date'] is None and before['data_status']=='INVALID_NO_DATA'
    warm=model.get_liquidity_features_as_of('2026-09-17')
    assert warm['feature_status']=='WARMUP_INSUFFICIENT_HISTORY' and warm['z_score'] is None


@pytest.mark.parametrize('xi',[[1,0,0],[0,0,1],[.2,.3,.5]])
@pytest.mark.parametrize('anchor',[None,1.,1e9])
def test_macro_assumptions_cannot_distort_quantiles(xi,anchor):
    result=InstitutionalMarkovEngine().condition_timesfm_quantiles(90,100,110,xi,
        structural_target=anchor,structural_weight=1.)
    assert [result[k] for k in ['reconciled_p10','reconciled_p50','reconciled_p90']]==[90,100,110]
    assert result['structural_weight']==0
    assert result['prediction_interval']['calibration_status']=='NOT_ESTABLISHED'
    assert not result['macro_interpretation']['scenario_anchor_applied_to_forecast']


@pytest.mark.parametrize('quantiles',[(110,100,90),(90,float('nan'),110),(-1,100,110)])
def test_bad_quantiles_rejected(quantiles):
    with pytest.raises(ValueError,match='INVALID_QUANTILES'):
        InstitutionalMarkovEngine().condition_timesfm_quantiles(*quantiles,[1,0,0])


def test_invalid_interval_never_earns_negative_score():
    with pytest.raises(ValueError):
        compute_winkler_interval_score(100,110,90)


def test_research_contract_exposes_runtime_and_missing_liquidity():
    from core.pipeline import finalize_research_outputs, create_parameter_record
    from core.reconciliation import ReconciliationEngine
    cond=InstitutionalMarkovEngine().condition_timesfm_quantiles(90,100,110,[1,0,0])
    allocation={'target_risk_weight':.8, 'tactical_action':'Buy'}
    prior={'engine':'M1_GEOMETRIC_FALLBACK','timesfm_executed':False,'degraded_mode':True}
    finalize_research_outputs(cond,prior,allocation,{
        'z_liq':create_parameter_record(None,'cache',feature_status='DATA_STALE_CRITICAL')})
    assert cond['central_forecast']['source']=='M1_GEOMETRIC_FALLBACK'
    assert cond['prediction_interval']['nominal_coverage'] is None
    assert cond['macro_interpretation']['liquidity_feature_status']=='DATA_STALE_CRITICAL'
    assert allocation['allocation_disabled'] and allocation['target_risk_weight']==0
    summary=ReconciliationEngine().generate_plain_english_summary('BTC',100,'$',cond,allocation,'old rule')
    assert 'DATA_STALE_CRITICAL' in summary and 'M1_GEOMETRIC_FALLBACK' in summary
    assert 'WHAT TO DO WITH YOUR MONEY' not in summary and 'Mechanism-Adjusted' not in summary


def test_legacy_evaluator_rejects_repaired_core_before_inference():
    from scripts.evaluate_btc_protocol_002 import main
    with pytest.raises(RuntimeError,match='ARCHIVED_EVALUATION'):
        main()
