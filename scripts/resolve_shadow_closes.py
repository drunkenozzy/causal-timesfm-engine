"""Resolve closed target sessions independently of new-forecast eligibility."""
import json
import math
from pathlib import Path


def resolve_closed_sessions(path, prices, today, ledger_factory):
    path = Path(path)
    ledger = json.loads(path.read_text()) if path.exists() else {}
    if prices.empty:
        return ledger
    events = None
    changed = False
    for day, entry in ledger.items():
        fc = entry.get('forecast', {})
        if day >= today or not fc or fc.get('actual_close') is not None:
            continue
        matches = prices.loc[prices.index.strftime('%Y-%m-%d') == day, 'Close']
        if len(matches) != 1 or not math.isfinite(float(matches.iloc[0])) or matches.iloc[0] <= 0:
            continue
        actual = float(matches.iloc[0])
        errors = {f'{m}_error': actual - fc[f'{m}_forecast'] if fc.get(f'{m}_forecast') is not None else None
                  for m in ('m1', 'm4')}
        if fc.get('forecast_id') and fc.get('experiment_status') not in ('PILOT_NOWCAST', 'LEDGER_RESOLUTION_ERROR'):
            events = events if events is not None else ledger_factory()
            # Recover safely if an earlier process appended the event but failed before saving display state.
            existing = events.list_resolutions(fc['forecast_id']) if hasattr(events, 'list_resolutions') else []
            if existing:
                if len(existing) != 1 or existing[0]['realized_actual'] != actual:
                    raise ValueError(f'Conflicting resolution for {day}')
            else:
                events.resolve_forecast(forecast_id=fc['forecast_id'], realized_actual=actual,
                                        secondary_metrics=errors, notes='Closed native provider session')
        fc.update(actual_close=actual, **errors)
        changed = True
        print(f'[Shadow] Scored closed session {day}: actual={actual}, errors={errors}')
    if changed:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(ledger, indent=4))
        temporary.replace(path)
    return ledger
