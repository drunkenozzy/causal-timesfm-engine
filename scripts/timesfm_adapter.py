import sys
import numpy as np

import timesfm as real_timesfm

class MockHparams:
    def __init__(self, backend, per_core_batch_size, horizon_len):
        pass

class MockCheckpoint:
    def __init__(self, huggingface_repo_id):
        self.huggingface_repo_id = huggingface_repo_id

class MockTimesFm:
    def __init__(self, hparams, checkpoint):
        print(f'[Adapter] Loading weights from {checkpoint.huggingface_repo_id} via TimesFM3Forecaster...')
        self.model = real_timesfm.TimesFM3Forecaster.from_pretrained(checkpoint.huggingface_repo_id)

    def forecast(self, inputs, freq):
        ctx = np.array(inputs[0])
        # engine_timesfm sets required_horizon=horizon_days. The actual horizon is len of freq?
        # Wait, core/engine_timesfm.py calls:
        # self._model.forecast(inputs=[vals], freq=[freq_indicator])
        # It doesn't pass horizon_days! The horizon is fixed at initialization in TimesFmHparams.
        # So we just use 90 days.
        horizon = 90
        
        res = self.model.predict(context=ctx, horizon=horizon, return_quantiles=True)
        
        # res.forecast shape: (horizon,) -> reshape to (1, horizon)
        point_forecast = res.forecast.reshape(1, horizon)
        
        # res.quantiles shape: (horizon, 9)
        # engine_timesfm expects p10 at index 1 and p90 at index 9.
        # We will map:
        # 0: p?, 1: p10, 2: p20, 3: p30, 4: p40, 5: p50, 6: p60, 7: p70, 8: p80, 9: p90
        # res.quantiles has 9 values. Assuming they are p10...p90.
        
        q_out = np.zeros((1, horizon, 10))
        for i in range(horizon):
            q_out[0, i, 1] = res.quantiles[i, 0] # p10
            q_out[0, i, 5] = res.quantiles[i, 4] # p50
            q_out[0, i, 9] = res.quantiles[i, 8] # p90
            
        return point_forecast, q_out

# Replace the module in sys.modules
sys.modules['timesfm'] = type('mock_timesfm', (object,), {
    'TimesFm': MockTimesFm,
    'TimesFmHparams': MockHparams,
    'TimesFmCheckpoint': MockCheckpoint
})

if __name__ == '__main__':
    from core.engine_timesfm import TimesFmBaselineEngine
    tfm = TimesFmBaselineEngine()
    res = tfm.forecast([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], horizon_days=1)
    print(res)
