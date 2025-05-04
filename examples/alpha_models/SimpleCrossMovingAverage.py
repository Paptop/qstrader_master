from qstrader.alpha_model.alpha_model import AlphaModel

class SimpleCrossMovingAverage(AlphaModel):
    def __init__(
        self,
        signals,
        universe,
        data_handler,
        long,
        short
    ):
        self.signals = signals
        self.universe = universe
        self.data_handler = data_handler
        self.long = long
        self.short = short
        self.weights = None
        self.isSelfInvested = False
        self.step_size = 0.05  # Step for gradual weight adjustment

    def get_description(self):
        return {
            "name": "SimpleCrossMovingAverage",
            "description": "Simple cross moving average strategy",
            "parameters": {
                "short": self.short,
                "long": self.long
            }
        }
    def __call__(self, dt, stats):
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

        short_sma = self.signals['sma'](assets[0], int(self.short))
        long_sma = self.signals['sma'](assets[0], int(self.long))
                
        if short_sma > long_sma and not self.isSelfInvested:
            self.weights[assets[0]] = 1.0
            self.weights[assets[1]] = 0.0
            self.isSelfInvested = True
            #print(dt, "Alpha model: long position for market")
        
        elif short_sma < long_sma and self.isSelfInvested:
            self.weights[assets[0]] = 0.0
            self.weights[assets[1]] = 1.0
            self.isSelfInvested = False
            #print(dt, "Alpha model: short position for market")        
        return self.weights