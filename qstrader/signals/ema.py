import math
import numpy as np

from qstrader.signals.signal import Signal

class EMASignal(Signal):
    """
    Indicator class to calculate exponential moving average (EMA)
    of the last N periods for a set of prices.

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

    def _exponential_moving_average(self, asset, lookback):
        """
        Calculate the 'trend' for the provided lookback
        period based on the exponential moving average (EMA)
        of the price buffers for a particular asset.

        Parameters
        ----------
        asset : `str`
            The asset symbol name.
        lookback : `int`
            The lookback period.

        Returns
        -------
        `float`
            The EMA value ('trend') for the period.
        """
        # Retrieve the stored price series for this asset and lookback
        prices = self.buffers.prices[f"{asset}_{lookback}"]
        
        # If we have no prices, return NaN or 0.0, depending on your preference
        if len(prices) == 0:
            return np.nan
        
        # Calculate the smoothing factor alpha
        #alpha = 1.0 - (0.5 ** (1.0 / lookback))
        alpha = 2.0 / (lookback + 1.0)
        #alpha = 1 - math.exp(-1.0 / lookback)

        # Initialize EMA with the first price
        ema = prices[0]

        # Iteratively update EMA for each subsequent price
        for price in list(prices)[1:]:
            ema = alpha * price + (1.0 - alpha) * ema

        return ema

    def __call__(self, asset, lookback):
        """
        Calculate the lookback-period trend
        for the asset (the EMA).

        Parameters
        ----------
        asset : `str`
            The asset symbol name.
        lookback : `int`
            The lookback period.

        Returns
        -------
        `float`
            The trend (EMA) for the period.
        """
        return self._exponential_moving_average(asset, lookback)
