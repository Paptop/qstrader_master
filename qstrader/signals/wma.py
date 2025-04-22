import numpy as np

from qstrader.signals.signal import Signal


class WMASignal(Signal):
    """
    Indicator class to calculate simple moving average
    of last N periods for a set of prices.

    Parameters
    ----------
    start_dt : `pd.Timestamp`
        The starting datetime (UTC) of the signal.
    universe : `Universe`
        The universe of assets to calculate the signals for.
    lookbacks : `list[int]`
        The number of lookback periods to store prices for.
    """

    def __init__(self, start_dt, universe, lookbacks):
        super().__init__(start_dt, universe, lookbacks)

    def _generate_decreasing_weights(self, prices):
        """
        Returns a NumPy array of weights that decrease from 1.0 to a positive value
        in equally spaced steps, matching the length of `prices`.
        
        Example for 4 prices:
            n = 4
            step = 1/4 = 0.25
            weights = [1.0, 0.75, 0.5, 0.25]
        """
        n = len(prices)
        step = 1 / n #0.25
        # For i in [0, 1, 2, ..., n-1], each weight = 1.0 - (i * step).
        weights = 1.0 - np.arange(n) * step
        weights = np.clip(weights, 0, None)
        return weights

    def _weighted_moving_average(self, asset, lookback):
        """
        Calculate the 'trend' for the provided lookback
        period based on the simple moving average of the
        price buffers for a particular asset.

        Parameters
        ----------
        asset : `str`
            The asset symbol name.
        lookback : `int`
            The lookback period.

        Returns
        -------
        `float`
            The WMA value ('trend') for the period.
        """
        #return np.mean(self.buffers.prices['%s_%s' % (asset, lookback)])
        prices = np.array(self.buffers.prices['%s_%s' % (asset, lookback)])
        weights = self._generate_decreasing_weights(prices)
        return np.average(prices, weights=weights)

    def __call__(self, asset, lookback):
        """
        Calculate the lookback-period trend
        for the asset.

        Parameters
        ----------
        asset : `str`
            The asset symbol name.
        lookback : `int`
            The lookback period.

        Returns
        -------
        `float`
            The trend (WMA) for the period.
        """
        return self._weighted_moving_average(asset, lookback)
