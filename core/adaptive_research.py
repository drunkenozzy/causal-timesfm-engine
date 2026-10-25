"""Local, append-only forecast correction. Never submits orders or promotes a model."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

CONFIG = dict(version='local-adaptive-v1', window=252, minimum_train=40,
              minimum_validation=20, validation_window=30, minimum_gain=.05,
              ridge_penalty=10., maximum_log_correction=.02)
# Exact definitions/units are fixed in docs/LOCAL_ADAPTIVE_RESEARCH.md.
BLOCKS = {
    'raw': ('dxy_return_1d', 'dxy_return_5d', 'us10y_change_5d_pp', 'brent_wti_spread_pct', 'target_return_5d'),
    'network': ('signed_dollar_shock', 'downside_connectivity'),
}
SOURCE_HASH = hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
RECIPE = hashlib.sha256(json.dumps([CONFIG, BLOCKS, SOURCE_HASH], sort_keys=True).encode()).hexdigest()


def utc(value):
    value = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if value.tzinfo is None:
        raise ValueError('Timezone required')
    return value.astimezone(timezone.utc)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def features(packet):
    """Reject future/stale/unversioned features; missing blocks remain unavailable."""
    result = {}
    for name, item in packet.get('features', {}).items():
        if name not in sum((list(v) for v in BLOCKS.values()), []):
            raise ValueError('Unregistered feature: ' + name)
        age = (utc(packet['decision_utc']) - utc(item['observed_at_utc'])).total_seconds()/86400
        if not (0 <= age <= 4) or utc(item['available_at_utc']) > utc(packet['decision_utc']):
            raise ValueError('Unavailable or stale feature: ' + name)
        if utc(item['available_at_utc']) < utc(item['observed_at_utc']):
            raise ValueError('Feature publication precedes observation')
        if not item.get('source') or not math.isfinite(float(item['value'])):
            raise ValueError('Feature needs finite value and source')
        result[name] = float(item['value'])
    return result


def estimate(packet, history):
    """Refit candidates on matured outcomes, then select on stored earlier predictions."""
    history = history[-CONFIG['window']:]
    p = packet['p50']
    current = features(packet)
    candidates = {'timesfm': p, 'persistence': packet['origin_close']}
    counts = {}
    if len(history) >= CONFIG['minimum_train']:
        residual = np.array([math.log(r['actual']/r['packet']['p50']) for r in history])
        cap = CONFIG['maximum_log_correction']
        candidates['bias'] = p * math.exp(float(np.clip(residual.mean(), -cap, cap)))
        counts['bias'] = len(history)
        for arm, names in [('raw', BLOCKS['raw']), ('network', BLOCKS['raw']+BLOCKS['network'])]:
            usable = [r for r in history if all(k in r['packet'].get('features', {}) for k in names)]
            if not all(k in current for k in names) or len(usable) < CONFIG['minimum_train']:
                continue
            x = np.array([[r['packet']['features'][k]['value'] for k in names] for r in usable], dtype=float)
            y = np.array([math.log(r['actual']/r['packet']['p50']) for r in usable])
            mu, scale = x.mean(axis=0), x.std(axis=0)
            scale[scale < 1e-8] = 1.
            x = (x-mu)/scale
            beta = np.linalg.solve(x.T@x + CONFIG['ridge_penalty']*np.eye(len(names)), x.T@(y-y.mean()))
            correction = float(y.mean() + ((np.array([current[k] for k in names])-mu)/scale)@beta)
            candidates[arm] = p*math.exp(float(np.clip(correction, -cap, cap)))
            counts[arm] = len(usable)
    selected = 'timesfm'
    checks = {}
    # Every comparison uses the same stored, previously issued candidate predictions.
    for arm in ['persistence', 'bias', 'raw', 'network']:
        if arm not in candidates:
            continue
        references = {'timesfm', 'persistence', selected}
        if arm == 'raw': references.add('bias')
        if arm == 'network': references.update(['bias', 'raw'])
        required = references | {arm}
        sample = [r for r in history if required <= r['candidates'].keys()][-CONFIG['validation_window']:]
        if len(sample) < CONFIG['minimum_validation']:
            checks[arm] = dict(status='WARMUP', count=len(sample))
            continue
        losses = {a: float(np.mean([((r['actual']-r['candidates'][a])/r['packet']['origin_close'])**2 for r in sample])) for a in required}
        others = references - {arm}
        passed = all(losses[arm] < (1-CONFIG['minimum_gain'])*losses[a] for a in others)
        checks[arm] = dict(status='PASS_SHADOW_ONLY' if passed else 'NO_GAIN', count=len(sample), losses=losses)
        if passed:
            selected = arm
    return dict(selected=selected, prediction=candidates[selected], candidates=candidates,
                training_counts=counts, validation=checks, production_changed=False,
                interval_for_selected_candidate=None,
                base_interval=dict(lower=packet['p10'], upper=packet['p90'], source='TimesFM', calibration='NOT_ESTABLISHED'),
                training_ids=[r['id'] for r in history], training_sha256=digest(history))


class LocalLearningLedger:
    """Single-writer journal. Call under one local process; replay uses a separate file."""
    def __init__(self, path, mode='PROSPECTIVE_SHADOW'):
        if mode not in ('PROSPECTIVE_SHADOW', 'RETROSPECTIVE_REPLAY'):
            raise ValueError('Unknown mode')
        self.path, self.mode = Path(path), mode
        self.disk_hash = hashlib.sha256(self.path.read_bytes() if self.path.exists() else b'').hexdigest()
        self.events = []
        self.predictions, self.resolutions = {}, {}
        for line in self.path.read_text(encoding='utf-8').splitlines() if self.path.exists() else []:
            event = json.loads(line)
            payload = {k: v for k, v in event.items() if k != 'hash'}
            previous = self.events[-1]['hash'] if self.events else 'GENESIS'
            if event['hash'] != digest(payload) or event['previous_hash'] != previous or event['mode'] != mode:
                raise ValueError('Journal integrity or mode mismatch')
            if event['kind'] == 'prediction':
                if event['id'] in self.predictions: raise ValueError('Duplicate prediction')
                self.predictions[event['id']] = event
            elif event['kind'] == 'resolution':
                if event['id'] not in self.predictions or event['id'] in self.resolutions:
                    raise ValueError('Invalid resolution')
                self.resolutions[event['id']] = event
            else:
                raise ValueError('Unknown event')
            self.events.append(event)

    def append(self, event):
        event = dict(event, mode=self.mode, previous_hash=self.events[-1]['hash'] if self.events else 'GENESIS')
        event['hash'] = digest(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_suffix(self.path.suffix+'.lock')
        lock_handle = lock.open('x')
        try:
            with lock_handle:
                content = self.path.read_bytes() if self.path.exists() else b''
                if hashlib.sha256(content).hexdigest() != self.disk_hash:
                    raise ValueError('Another writer changed the journal; reload before retrying')
                with self.path.open('a', encoding='utf-8', newline='\n') as handle:
                    handle.write(json.dumps(event, sort_keys=True, allow_nan=False)+'\n')
                self.disk_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        finally:
            lock.unlink()
        self.events.append(event)
        (self.predictions if event['kind']=='prediction' else self.resolutions)[event['id']] = event
        return event

    def register(self, packet, now=None):
        now = utc(now or datetime.now(timezone.utc).isoformat())
        decision, terminal = utc(packet['decision_utc']), utc(packet['terminal_utc'])
        if decision > now or now >= terminal or (self.mode=='PROSPECTIVE_SHADOW' and (now-decision).total_seconds()>300):
            raise ValueError('Prediction must be registered at decision time before terminal')
        if not packet.get('asset') or not packet.get('source') or not packet.get('timesfm_executed'):
            raise ValueError('Asset, source and genuine TimesFM forecast required')
        if packet.get('degraded_mode', True): raise ValueError('Degraded neural forecast')
        if packet['horizon_days'] not in (1, 7): raise ValueError('Supported horizons: 1 or 7 days')
        origin = utc(packet['origin_utc'])
        if not origin <= decision or terminal-origin != timedelta(days=packet['horizon_days']):
            raise ValueError('Horizon does not match the origin and terminal closes')
        if not all(math.isfinite(packet[k]) and packet[k]>0 for k in ['origin_close','p10','p50','p90']):
            raise ValueError('Invalid price')
        if not packet['p10']<=packet['p50']<=packet['p90']: raise ValueError('Unordered quantiles')
        features(packet)
        key = digest([packet['asset'], packet['horizon_days'], terminal.isoformat(), RECIPE])
        if key in self.predictions:
            old = self.predictions[key]
            if old['packet'] != packet: raise ValueError('Cannot replace a registered prediction')
            return old
        history = []
        for identifier, p in self.predictions.items():
            r = self.resolutions.get(identifier)
            if (r and p['recipe'] == RECIPE and p['packet']['asset'] == packet['asset']
                    and p['packet']['horizon_days'] == packet['horizon_days']
                    and utc(p['packet']['terminal_utc']) <= decision
                    and utc(r['recorded_at_utc']) <= decision):
                history.append(dict(id=identifier, packet=p['packet'], actual=r['actual'], candidates=p['result']['candidates']))
        history.sort(key=lambda r: utc(r['packet']['decision_utc']))
        result = estimate(packet, history)
        return self.append(dict(kind='prediction', id=key, recipe=RECIPE, packet=packet,
                                recorded_at_utc=now.isoformat(), result=result))

    def resolve(self, identifier, actual, source, now=None):
        now = utc(now or datetime.now(timezone.utc).isoformat())
        original = self.predictions[identifier]
        if now < utc(original['packet']['terminal_utc']): raise ValueError('Outcome has not matured')
        if not math.isfinite(actual) or actual<=0 or not source: raise ValueError('Invalid sourced actual')
        if identifier in self.resolutions:
            old = self.resolutions[identifier]
            if old['actual'] != actual or old['source'] != source: raise ValueError('Actual revision requires separate review')
            return old
        return self.append(dict(kind='resolution', id=identifier, actual=float(actual), source=source,
                                recorded_at_utc=now.isoformat()))
