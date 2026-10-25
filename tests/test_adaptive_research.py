from datetime import datetime, timedelta, timezone
import json

import pytest
from core.adaptive_research import BLOCKS, LocalLearningLedger, estimate


def packet(day=0,asset='BTC-USD',h=1,p50=104):
    decision=datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(days=day)
    return dict(asset=asset,horizon_days=h,decision_utc=decision.isoformat(),
                origin_utc=decision.isoformat(),
                terminal_utc=(decision+timedelta(days=h)).isoformat(),origin_close=100.,
                p10=90.,p50=float(p50),p90=115.,features={},source='unit-test',
                timesfm_executed=True,degraded_mode=False)


def add_features(p,sign=1):
    for key in BLOCKS['raw']:
        p['features'][key]=dict(value=sign*.01 if key=='dxy_return_5d' else 0.,
                               observed_at_utc=p['decision_utc'],available_at_utc=p['decision_utc'],source='test')
    return p


def test_persistent_learning_refits_and_corrects_bias_only_after_validation(tmp_path):
    path=tmp_path/'journal.jsonl'
    learner=LocalLearningLedger(path,'RETROSPECTIVE_REPLAY')
    for i in range(85):
        p=packet(i)
        if i: learner.resolve(old['id'],102.,'test',now=p['decision_utc'])
        old=learner.register(p,now=p['decision_utc'])
        if i<40: assert 'bias' not in old['result']['candidates']
        if i<60: assert old['result']['selected']!='bias'
    assert old['result']['selected']=='bias'
    assert abs(old['result']['prediction']-102)<.01
    reloaded=LocalLearningLedger(path,'RETROSPECTIVE_REPLAY')
    assert reloaded.predictions[old['id']]==old
    assert old['result']['interval_for_selected_candidate'] is None


def test_raw_features_can_be_tested_without_being_forced_into_forecast(tmp_path):
    learner=LocalLearningLedger(tmp_path/'features.jsonl','RETROSPECTIVE_REPLAY')
    for i in range(90):
        p=add_features(packet(i,p50=100),1 if i%2 else -1)
        if i: learner.resolve(old['id'],101. if (i-1)%2 else 99.,'test',now=p['decision_utc'])
        old=learner.register(p,now=p['decision_utc'])
    assert old['result']['selected']=='raw'
    assert 'network' not in old['result']['candidates']
    assert not old['result']['production_changed']


def test_overlapping_targets_and_asset_horizon_isolation(tmp_path):
    learner=LocalLearningLedger(tmp_path/'ledger','RETROSPECTIVE_REPLAY')
    p=packet(0,h=7);first=learner.register(p,now=p['decision_utc'])
    with pytest.raises(ValueError,match='not matured'):
        learner.resolve(first['id'],102.,'test',now=packet(1)['decision_utc'])
    learner.resolve(first['id'],102.,'test',now=packet(7)['decision_utc'])
    for asset,h in [('BTC-USD',1),('ETH-USD',7)]:
        p=packet(8,asset,h);result=learner.register(p,now=p['decision_utc'])
        assert result['result']['training_ids']==[]
    p=packet(8,h=7);result=learner.register(p,now=p['decision_utc'])
    assert result['result']['training_ids']==[first['id']]


def test_late_resolution_is_not_available_to_an_earlier_decision(tmp_path):
    learner=LocalLearningLedger(tmp_path/'ledger','RETROSPECTIVE_REPLAY')
    p=packet();first=learner.register(p,now=p['decision_utc'])
    learner.resolve(first['id'],102.,'test',now=packet(10)['decision_utc'])
    p=packet(5);result=learner.register(p,now=p['decision_utc'])
    assert not result['result']['training_ids']


@pytest.mark.parametrize('issue',['future','stale','unknown','nan'])
def test_bad_explanatory_inputs_fail_closed(tmp_path,issue):
    p=add_features(packet(10))
    f=p['features']['dxy_return_5d']
    if issue=='future': f['available_at_utc']=packet(11)['decision_utc']
    if issue=='stale': f['observed_at_utc']=packet(0)['decision_utc']
    if issue=='unknown': p['features']['narrative_sentiment']=p['features'].pop('dxy_return_5d')
    if issue=='nan': f['value']=float('nan')
    with pytest.raises(ValueError):
        LocalLearningLedger(tmp_path/'l','RETROSPECTIVE_REPLAY').register(p,now=p['decision_utc'])


def test_changed_predictions_actuals_and_replay_as_live_are_rejected(tmp_path):
    path=tmp_path/'l';l=LocalLearningLedger(path,'RETROSPECTIVE_REPLAY')
    p=packet();r=l.register(p,now=p['decision_utc'])
    assert l.register(p,now=p['decision_utc'])==r
    changed=dict(p,p50=105.)
    with pytest.raises(ValueError,match='replace'):l.register(changed,now=p['decision_utc'])
    l.resolve(r['id'],102.,'test',now=p['terminal_utc'])
    with pytest.raises(ValueError,match='revision'):l.resolve(r['id'],103.,'test',now=p['terminal_utc'])
    with pytest.raises(ValueError,match='mode mismatch'):LocalLearningLedger(path)
    with pytest.raises(ValueError,match='decision time'):
        LocalLearningLedger(tmp_path/'live').register(p,now=packet(1)['decision_utc'])


def test_tampering_and_stale_concurrent_writer_fail_closed(tmp_path):
    path=tmp_path/'l';a=LocalLearningLedger(path,'RETROSPECTIVE_REPLAY');b=LocalLearningLedger(path,'RETROSPECTIVE_REPLAY')
    p=packet();a.register(p,now=p['decision_utc'])
    p=packet(1)
    with pytest.raises(ValueError,match='Another writer'):b.register(p,now=p['decision_utc'])
    assert not path.with_suffix('.lock').exists()
    data=json.loads(path.read_text());data['result']['prediction']=999
    path.write_text(json.dumps(data)+'\n')
    with pytest.raises(ValueError,match='integrity'):LocalLearningLedger(path,'RETROSPECTIVE_REPLAY')


def test_horizon_and_late_registration_cannot_be_mislabelled(tmp_path):
    p=packet(h=7);p['terminal_utc']=packet(h=1)['terminal_utc']
    with pytest.raises(ValueError,match='Horizon'):
        LocalLearningLedger(tmp_path/'a').register(p,now=p['decision_utc'])
    p=packet();p['decision_utc']='2026-01-01T23:59:00+00:00'
    with pytest.raises(ValueError,match='decision time'):
        LocalLearningLedger(tmp_path/'b').register(p,now='2026-01-02T00:00:01+00:00')
