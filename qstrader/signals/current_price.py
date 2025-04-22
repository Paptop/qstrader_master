import numpy as np

from qstrader.signals.signal import Signal

class CurrentPriceSignal(Signal):

    def __init__(self, start_dt, universe, lookbacks):
        super().__init__(start_dt, universe, lookbacks)

    def __call__(self, asset, lookback):
        return self.buffers.prices['%s_%s' % (asset, lookback)]
