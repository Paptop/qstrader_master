import math
import numpy as np

from qstrader.signals.signal import Signal

class KaufmanAdaptiveMASignal(Signal):
    """
    Indicator class to calculate Kaufman's Adaptive Moving Average (KAMA),
    sometimes known as AMA, of the last N periods for a set of prices.
    
    This uses a dynamic smoothing factor 'alpha' that depends on the
    'efficiency ratio' (ER), which measures how trending vs. noisy
    the price series is.

    Parameters
    ----------
    start_dt : `pd.Timestamp`
        The starting datetime (UTC) of the signal.
    universe : `Universe`
        The universe of assets to calculate the signals for.
    lookbacks : `list[int]`
        The lookback periods to store prices for. This might be used
        differently here than a simple EMA; we'll still store them 
        for compatibility.
    fast_period : int
        Period used for the 'fast' smoothing factor in KAMA.
    slow_period : int
        Period used for the 'slow' smoothing factor in KAMA.
    er_period : int
        The period over which to compute the efficiency ratio (ER).
        Typically smaller than slow_period (e.g., 10).
    """

    def __init__(
        self, 
        start_dt, 
        universe, 
        lookbacks,
        fast_period=2, 
        slow_period=30, 
        er_period=10
    ):
        super().__init__(start_dt, universe, lookbacks)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.er_period = er_period

        # Precompute the "fast" and "slow" SC factors 
        # (the base of alpha range)
        #self.fast_alpha = 2.0 / (self.fast_period + 1.0)
        #self.slow_alpha = 2.0 / (self.slow_period + 1.0)

        self.fast_alpha = 1.0 - (0.5 ** (1.0 / self.fast_period))
        self.slow_alpha = 1.0 - (0.5 ** (1.0 / self.slow_period))

    def _kaufman_ama(self, asset, lookback):
        """
        Calculate Kaufman's Adaptive Moving Average for the given asset
        price buffer and return the final AMA value.

        Parameters
        ----------
        asset : `str`
            The asset symbol name.
        lookback : `int`
            The lookback period for which we store data in the 
            buffers. In typical usage, this might be >= slow_period 
            or something else. We'll still just retrieve the entire 
            price history we have for that asset/lookback combo.
        
        Returns
        -------
        float
            The final KAMA value for that price series.
        """
        # Retrieve the stored price series for this asset/lookback
        prices = self.buffers.prices.get(f"{asset}_{lookback}", [])
        if len(prices) == 0:
            return np.nan  # or 0.0 if you prefer

        # Convert to a list or NumPy array for convenience
        prices = np.array(prices, dtype=float)

        # Initialize the AMA with the first price (could also do
        # an average of the first 'er_period' prices).
        kama = prices[0]

        # For each new price from index=1 onward, compute an adaptive alpha.
        for i in range(1, len(prices)):
            # We'll only compute ER if i >= er_period
            if i >= self.er_period:
                # Price movement ("signal")
                signal = abs(prices[i] - prices[i - self.er_period])
                
                # Summation of absolute price changes ("noise")
                noise = np.sum(np.abs(prices[j] - prices[j - 1]) 
                               for j in range(i - self.er_period + 1, i + 1))
            else:
                # If we don't have enough history, 
                # just treat ER as 0 or skip. 
                # We'll do a simple fallback: ER=1 for the very early bars
                # or you might do something more sophisticated.
                signal = 0.0
                noise = 1e-9  # avoid zero-division
                          
            er = signal / noise if noise != 0 else 0.0

            # Compute the SC (smoothing constant)
            sc = (er * (self.fast_alpha - self.slow_alpha) + self.slow_alpha)**2

            # Update the KAMA
            kama = kama + sc * (prices[i] - kama)

        return kama

    def __call__(self, asset, lookback):
        """
        Calculate the lookback-period KAMA for the asset.

        Parameters
        ----------
        asset : `str`
            The asset symbol name.
        lookback : `int`
            The lookback period. This is used to fetch the appropriate 
            price buffer, but the actual KAMA parameters (er_period, 
            fast_period, slow_period) are part of this class instance.
        
        Returns
        -------
        float
            The final KAMA value for the period.
        """
        return self._kaufman_ama(asset, lookback)
