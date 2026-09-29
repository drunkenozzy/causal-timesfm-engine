"""Past-only, horizon-specific calibration for the ETH spot paper study."""
import numpy as np
import pandas as pd

LEVELS = np.array([.1, .5, .9])
CAL_WINDOW = 252


def calibrate(records, daily, window=CAL_WINDOW):
    """Rescale past median errors by each forecast's own log interquantile width.

    No normality assumption and no use of today's or unfinished outcomes. Three
    quantiles of completed standardized errors shift/scale the current bracket.
    """
    dates = [pd.Timestamp(r['date'], tz='UTC') for r in records]
    if dates != sorted(set(dates)):
        raise ValueError('Forecast dates must be unique and ordered')
    output = []
    for i, row in enumerate(records):
        date = dates[i]
        item = {**row, 'calibrated': {}, 'calibration_audit': {}}
        for horizon in [1, 2, 3]:
            key = str(horizon)
            raw = np.asarray(row['raw'][key], dtype=float)
            if raw.shape != (3,) or not np.isfinite(raw).all() or raw[0] <= 0 or not (raw[0] < raw[1] < raw[2]):
                raise ValueError('Invalid or crossed raw quantiles; no substitute allowed')
            candidates = [j for j in range(i) if dates[j] + pd.Timedelta(days=horizon) <= date][-window:]
            if len(candidates) < window:
                continue
            errors = []
            for j in candidates:
                target = dates[j] + pd.Timedelta(days=horizon - 1)
                actual = float(daily.loc[target, 'close'])
                q = np.log(np.asarray(records[j]['raw'][key], dtype=float))
                errors.append((np.log(actual) - q[1]) / (q[2] - q[0]))
            transformed = np.quantile(errors, LEVELS, method='linear')
            logq = np.log(raw)
            calibrated = np.exp(logq[1] + (logq[2] - logq[0]) * transformed)
            if not np.isfinite(calibrated).all() or not np.all(np.diff(calibrated) > 0):
                raise ValueError('Invalid calibrated quantiles')
            item['calibrated'][key] = calibrated.tolist()
            item['calibration_audit'][key] = {'completed_outcomes': len(candidates),
                'last_outcome_available': (dates[candidates[-1]] + pd.Timedelta(days=horizon)).isoformat()}
        output.append(item)
    return output


def coverage_report(records, daily, start='2025-01-01', end='2026-09-01'):
    selected = [r for r in records if start <= r['date'] < end]
    result = {}
    for h in [1, 2, 3]:
        key = str(h)
        actual = np.array([daily.loc[pd.Timestamp(r['date'], tz='UTC') + pd.Timedelta(days=h - 1), 'close'] for r in selected])
        result[key] = {}
        for family in ['raw', 'calibrated']:
            q = np.array([r[family][key] for r in selected])
            below = (actual[:, None] <= q).mean(axis=0)
            covered = (actual >= q[:, 0]) & (actual <= q[:, 2])
            residual = actual[:, None] - q
            pinball = np.maximum(LEVELS * residual, (LEVELS - 1) * residual).mean(axis=0)
            result[key][family] = {'origins': len(actual), 'below_p10_p50_p90': below.tolist(),
                'terminal_p10_p90_coverage': float(covered.mean()),
                'mean_bracket_width_pct': float(np.mean((q[:, 2] - q[:, 0]) / actual) * 100),
                'pinball_loss': pinball.tolist(),
                'descriptive_coverage_check': bool(.75 <= covered.mean() <= .85 and abs(below[0] - .1) <= .05 and abs(below[2] - .9) <= .05),
                'years': {year: float(covered[[r['date'].startswith(year) for r in selected]].mean())
                          for year in sorted({r['date'][:4] for r in selected})}}
    return result
