import pandas as pd
from qstrader.alpha_model.alpha_model import AlphaModel

class KamaModel(AlphaModel):
    def __init__(
        self,
        signals,
        universe,
        data_handler,
        period,
        #prices
    ):
        self.signals = signals
        self.universe = universe
        self.data_handler = data_handler
        self.weights = None
        self.isSelfInvested = False
        self.period = period
        #self.prices = prices
        #self.prices.index = pd.to_datetime(self.prices.index)

    def get_description(self):
        return {
            "name": "KamaModel",
            "description": "Kaufman signal strategy",
            "parameters": {
                "period": self.period
            }
        }
    def __call__(self, dt):
        """
        Produce the dictionary of fixed scalar signals for
        each of the Asset instances within the Universe.

        Parameters
        ----------
        dt : `pd.Timestamp`
            The time 'now' used to obtain appropriate data and universe
            for the the signals.

        Returns
        -------
        `dict{str: float}`
            The Asset symbol keyed scalar-valued signals.
        """

        assets = self.universe.get_assets(dt)

        if self.weights is None:
            self.weights = {asset: 0.0 for asset in assets}
            return self.weights
        
        kama_value = self.signals['kama'](assets[0], self.period)
        close_price_sp = self.signals['current_price'](assets[0], 1)[0]
        dd = dt.date()
        current_price = close_price_sp #self.get_last_existing_price(dd)

        # 3) Compare price to KAMA to decide weighting
        if current_price > kama_value:
            self.weights[assets[0]] = 1.0
            self.weights[assets[1]] = 0.0
            self.isSelfInvested = True
            print('Long position')
        else:
        # Price is below KAMA => Flat or short. Let's do flat:
            self.weights[assets[0]] = 0.0
            self.weights[assets[1]] = 1.0
            self.isSelfInvested = False
            print('Short position')

        return self.weights
    

    def get_last_existing_price(self, dd):
        """
        Attempt to retrieve the 'EQ:SPY' price on date `dd`.
        If it doesn't exist, return the most recent prior price.
        """
        dd_str = str(dd)  # Convert to string if your index is stored as strings

        try:
            # Try direct lookup first
            return self.prices.loc[dd_str, 'EQ:SPY']
        except KeyError:
            # That date doesn't exist or column not found
            pass

        # We get here if direct lookup failed. Now find the last date <= dd
        # 1) Filter rows up to dd_str (inclusive)
        # 2) Take the last row
        mask = self.prices.index <= dd_str
        df_filtered = self.prices.loc[mask, 'EQ:SPY']
        if len(df_filtered) == 0:
            # No earlier data at all
            return None

        # The last price prior to dd
        return df_filtered.iloc[-1]