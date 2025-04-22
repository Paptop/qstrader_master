import os
import sys

import numpy as np
import pandas as pd
import pytz
from datetime import datetime

# Get the absolute path to the cloned library
lib_path = os.path.abspath("./")  # Adjust the path as needed

# Add to sys.path if not already present
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from examples.data_loaders.data_loader import SimpleDataLoader
from examples.StrategyRunner import StrategyRunner

def create_risk_model(factory, start_dt, end_dt):
    params = {
         "smooth": False,
         "easing": False,
         "lag": False,
         "current_day_only": True,
         "random": False,
         "smooth_days": 10,  # Default 19
         "lag_in_days": 0,  # Default 0
         "window_size": 365  # Default 365
    }

    data_loader = SimpleDataLoader()

    # Get back in time
    start_hmm_dt = pd.Timestamp('1995-01-01 14:30:00', tz=pytz.UTC)
    end_hmm_dt = start_dt
    data_for_risk_model = data_loader.get_data_frame(start_hmm_dt, end_dt)

    features = {
        'Adj_Close_SPY_change': { 'w': 1.0, 'sort_by_returns': True, 'flip_values': False },
        #'Volume_SPY_change':  { 'w': 0.1, 'sort_by_returns': True, 'flip_values': True },
        #'Adj_Close_AAPL_change':  { 'w': 1.0, 'sort_by_returns': True }
        #'Volume_AAPL_change':  { 'w': 1.0, 'sort_by_returns': False }
        'Adj_Close_VIX_change':  { 'w': 1.0, 'sort_by_returns': False, 'flip_values': False },
        #'10Y_Treasury':  { 'w': 0.01, 'sort_by_returns': True, 'flip_values': False },
        #'2Y_Treasury':  { 'w': 0.01, 'sort_by_returns': True, 'flip_values': False },
        #'CPI': { 'w': 0.5, 'sort_by_returns': False, 'flip_values': True }
    }
    return factory.create_risk_model(start_hmm_dt, end_hmm_dt, data_for_risk_model, features, params)

if __name__ == "__main__":
    start_dt = pd.Timestamp('2015-01-01 14:30:00', tz=pytz.UTC)
    end_dt = pd.Timestamp('2025-02-28 23:59:00', tz=pytz.UTC)
    symbols = ['SPY', 'AGG']
    strategy_runner = StrategyRunner(start_dt, end_dt, symbols, "Runner1.0")
    risk_model = create_risk_model(strategy_runner.get_strategy_factory(), start_dt, end_dt)

    #print(strategy_runner.execute_custom_cma(10, 90, "weekly"))
    #print(strategy_runner.execute_custom_cma(10, 90, "weekly", risk_model))

    #print(strategy_runner.execute_ema(7, 8, "weekly"))
    #print(strategy_runner.execute_sma(7, 8, "weekly"))
    #print(strategy_runner.execute_sma(7, 8, "weekly", risk_model))

    best_short = 0
    best_long = 0
    best_profit_factor = 0

    for rebalance in ["end_of_month", "weekly", "daily"]:
        for short_window in range(5, 50):
            for long_window in range(short_window + 1, 201):
                res = strategy_runner.execute_ema(short_window, long_window, rebalance)
                profitFactor = res["ProfitFactor"]

                if profitFactor > best_profit_factor:
                    print("---Best EMA ", rebalance, "--")
                    best_profit_factor = profitFactor
                    best_short = short_window
                    best_long = long_window
                    print("Long: ", best_long)
                    print("Short: ", best_short)
                    print("Profit factor: ", best_profit_factor) 
        best_profit_factor = 0
    #print(strategy_runner.execute_sma(10, 90, "daily", risk_model))

    #print(strategy_runner.execute_wma(7, 8, "weekly"))
    #print(strategy_runner.execute_wma(7, 8, "weekly", risk_model))