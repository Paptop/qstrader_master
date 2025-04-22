from examples.alpha_models.CrossExponentialMovingAverage import CrossExponentialMovingAverage
from examples.alpha_models.CrossCustomMovingAverage import CrossCustomMovingAverage
from examples.alpha_models.CrossWeightedMovingAverage import CrossWMovingAverage
from examples.alpha_models.KamaModel import KamaModel
from examples.alpha_models.RsiModel import RSIBasedAlphaModel
from examples.alpha_models.SimpleCrossMovingAverage import SimpleCrossMovingAverage
from examples.risk_filters.MultiRiskHMMFilter import MultiRiskHMMFilter


class StrategyFactory:
    def __init__(self, signals, strategy_universe, strategy_data_handler):
        self.signals = signals
        self.strategy_universe = strategy_universe
        self.strategy_data_handler = strategy_data_handler

    def create_risk_model(self, start_hmm_dt, end_hmm_dt, complete_data, features, params):        
        return MultiRiskHMMFilter(
            universe=self.strategy_universe,
            data=complete_data,
            features=features,
            params=params,
            start_hmm_dt=start_hmm_dt,
            end_hmm_dt=end_hmm_dt
        )

    def create_strategy(self, strategy_type, **kwargs):
        if strategy_type == "cross_moving_average":
            return CrossCustomMovingAverage(
                self.signals,
                self.strategy_universe,
                self.strategy_data_handler,
                kwargs.get("long"),
                kwargs.get("short")
            )

        elif strategy_type == "cross_weighted_moving_average":
            return CrossWMovingAverage(
                self.signals,
                self.strategy_universe,
                self.strategy_data_handler,
                kwargs.get("long"),
                kwargs.get("short")
            )

        elif strategy_type == "cross_exponential_moving_average":
            return CrossExponentialMovingAverage(
                self.signals,
                self.strategy_universe,
                self.strategy_data_handler,
                kwargs.get("long"),
                kwargs.get("short")
            )

        elif strategy_type == "rsi_based":
            return RSIBasedAlphaModel(
                self.signals,
                self.strategy_universe,
                self.strategy_data_handler,
                kwargs.get("rsi_period"),
                rsi_upper=kwargs.get("rsi_upper", 70),
                rsi_lower=kwargs.get("rsi_lower", 30)
            )
        
        elif strategy_type == "kama":
            return KamaModel(
                self.signals,
                self.strategy_universe,
                self.strategy_data_handler,
                self.kama_period,
                prices_for_kama=kwargs.get("prices_for_kama")
             )
        
        elif strategy_type == "simple_cross_moving_average":
            return SimpleCrossMovingAverage(
                self.signals,
                self.strategy_universe,
                self.strategy_data_handler,
                kwargs.get("long"),
                kwargs.get("short")
            )
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
