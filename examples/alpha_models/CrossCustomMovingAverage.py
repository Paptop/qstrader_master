from qstrader.alpha_model.alpha_model import AlphaModel

class CrossCustomMovingAverage(AlphaModel):
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
            "name": "CustomCrossMovingAverage",
            "description": "Custom cross moving average with exp strategy between assets",
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

        short_sma_sp = self.signals['ema'](assets[0], int(self.short))
        long_sma_sp = self.signals['ema'](assets[0], int(self.long))

        short_sma_b = self.signals['ema'](assets[1], int(self.short))
        long_sma_b = self.signals['ema'](assets[1], int(self.long))

        close_price_sp = self.signals['cur_price'](assets[0], 1)[0]
        close_price_b = self.signals['cur_price'](assets[1], 1)[0]

        rs_long = long_sma_sp / long_sma_b
        rs_short = short_sma_sp / short_sma_b
        rs_current = close_price_sp / close_price_b
        #trend_filter = rs_short > rs_long

        bull = (close_price_sp / close_price_b <= short_sma_sp/short_sma_b) #and trend_filter
        bear = (close_price_sp / close_price_b > short_sma_sp/short_sma_b) #and not trend_filter

        if bull and not self.isSelfInvested:
            self.weights[assets[0]] = 1.0
            self.weights[assets[1]] = 0.0
            self.isSelfInvested = True
            #print(dt, "Alpha model: long position for market")

        if bear and self.isSelfInvested:
            self.weights[assets[0]] = 0.0
            self.weights[assets[1]] = 1.0
            self.isSelfInvested = False
            #print(dt, "Alpha model: short position for market")

        #short_wma = self.signals['wma'](assets[0], int(self.short))
        #long_wma = self.signals['wma'](assets[0], int(self.long))


                
        #if short_sma > long_sma and not self.isSelfInvested:
            #self.weights[assets[0]] = min(1.0, self.weights[assets[0]] + self.step_size)
            #self.weights[assets[1]] = max(0.0, 1.0 - self.weights[assets[0]]) 
            #  # Increase allocation
            #self.weights[assets[1]] =  # Reduce asset[1]
            #self.weights[assets[0]] = 1.0
            #self.weights[assets[1]] = 0.0
            #self.isSelfInvested = True
            #print(dt, "Alpha model: long position for market")
        
        #elif short_sma < long_sma and self.isSelfInvested:
            #self.weights[assets[0]] = 0.0
            #self.weights[assets[1]] = 0.0
            #self.weights[assets[0]] = max(0.0, self.weights[assets[0]] - self.step_size)  # Decrease allocation
            #self.weights[assets[1]] = min(1.0, 1.0 - self.weights[assets[0]])  # Increase asset[1]
            #self.weights[assets[0]] = 0.0
            #self.weights[assets[0]] = 0.0
            #self.weights[assets[1]] = 1.0
            #self.isSelfInvested = False
            #print(dt, "Alpha model: short position for market")

        #self.weights['USD'] = 100
        
        return self.weights