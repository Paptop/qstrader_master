from qstrader.alpha_model.alpha_model import AlphaModel

class RSIBasedAlphaModel(AlphaModel):
    def __init__(
        self,
        signals,
        universe,
        data_handler,
        rsi_period,
        rsi_upper=70,
        rsi_lower=30
    ):
        self.signals = signals
        self.universe = universe
        self.data_handler = data_handler
        self.rsi_period = rsi_period
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower
        self.weights = None
        self.isSelfInvested = False

    def get_description(self):
        return {
            "name": "RSIBasedAlphaModel",
            "description": "RSI-based long/short strategy using fixed RSI thresholds",
            "parameters": {
                "rsi_period": self.rsi_period,
                "rsi_upper": self.rsi_upper,
                "rsi_lower": self.rsi_lower
            }
        }

    def __call__(self, dt):
        """
        Generate long/short weights based on RSI thresholds.

        Parameters
        ----------
        dt : `pd.Timestamp`
            The current datetime.

        Returns
        -------
        `dict{str: float}`
            Asset symbol keys with corresponding portfolio weights.
        """
        assets = self.universe.get_assets(dt)

        if self.weights is None:
            self.weights = {asset: 0.0 for asset in assets}
            return self.weights

        rsi = self.signals['rsi'](assets[0], int(self.rsi_period))

        if rsi < self.rsi_lower and not self.isSelfInvested:
            self.weights[assets[0]] = 1.0
            self.weights[assets[1]] = 0.0
            self.isSelfInvested = True
            print(dt, f"Alpha model: RSI {rsi:.2f} — LONG position for {assets[0]}")

        elif rsi > self.rsi_upper and self.isSelfInvested:
            self.weights[assets[0]] = 0.0
            self.weights[assets[1]] = 1.0
            self.isSelfInvested = False
            print(dt, f"Alpha model: RSI {rsi:.2f} — SHORT position for {assets[0]}")

        return self.weights
