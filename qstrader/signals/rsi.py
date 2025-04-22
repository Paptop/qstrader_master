import numpy as np
from qstrader.signals.signal import Signal


class RSISignal(Signal):
    """
    Signal class to calculate the Relative Strength Index (RSI)
    for a given asset over a specified lookback period.

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

    def _rsi(self, prices, period):
        """
        Compute the RSI value from price series.

        Parameters
        ----------
        prices : `np.ndarray`
            Array of historical prices.
        period : `int`
            Lookback period for RSI calculation.

        Returns
        -------
        `float`
            The RSI value.
        """
        if len(prices) < period:
            return np.nan  # Not enough data

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0  # Avoid division by zero

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def __call__(self, asset, lookback):
        """
        Compute RSI for the specified asset and lookback period.

        Parameters
        ----------
        asset : `str`
            The asset symbol.
        lookback : `int`
            The RSI lookback period.

        Returns
        -------
        `float`
            RSI value between 0 and 100.
        """
        prices = np.array(self.buffers.prices[f"{asset}_{lookback}"])
        return self._rsi(prices, lookback)
