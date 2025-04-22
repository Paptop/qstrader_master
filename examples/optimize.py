

from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

import numpy as np
import pandas as pd
import pytz
import json
import os
import sys
from datetime import datetime

# Get the absolute path to the cloned library
lib_path = os.path.abspath("./")  # Adjust the path as needed

# Add to sys.path if not already present
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

# Local libs
from examples.alpha_models.SimpleCrossMovingAverage import SimpleCrossMovingAverage
from examples.risk_filters.MultiRiskHMMFilter import MultiRiskHMMFilter
from examples.data_loaders.data_loader import SimpleDataLoader
from examples.alpha_models.RsiModel import RSIBasedAlphaModel
from examples.alpha_models.KamaModel import KamaModel
from qstrader.signals.rsi import RSISignal
from qstrader.signals.current_price import CurrentPriceSignal
from qstrader.signals.kama import KaufmanAdaptiveMASignal
from examples.alpha_models.CrossExponentialMovingAverage import CrossExponentialMovingAverage
from examples.alpha_models.CrossWeightedMovingAverage import CrossWMovingAverage
from examples.plots.plot import plot_sp500_with_regime
from examples.risk_filters.HMM import train_hmm, train_hmm_on_data
from examples.risk_filters.RiskHMMFilter import RiskHMMFilter
from examples.alpha_models.CrossCustomMovingAverage import CrossCustomMovingAverage

from qstrader.broker.fee_model.percent_fee_model import PercentFeeModel
from qstrader.signals.sma import SMASignal
from qstrader.alpha_model.fixed_signals import FixedSignalsAlphaModel
from qstrader.asset.equity import Equity
from qstrader.asset.universe.dynamic import DynamicUniverse
from qstrader.asset.universe.static import StaticUniverse
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.signals.signals_collection import SignalsCollection
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.backtest import BacktestTradingSession
from qstrader.signals.wma import WMASignal
from qstrader.signals.ema import EMASignal
def evaluate_combination(params):
    short_window, long_window, start_dt, end_dt, strategy_universe, strategy_data_handler, rebalance_freq = params
    
    try:
        sma = SMASignal(start_dt, strategy_universe, lookbacks=[short_window, long_window])
        signals = SignalsCollection({'sma': sma}, strategy_data_handler)

        alpha_model = SimpleCrossMovingAverage(
            signals,
            strategy_universe,
            strategy_data_handler,
            long_window,
            short_window
        )

        backtest = BacktestTradingSession(
            start_dt,
            end_dt,
            universe=strategy_universe,
            alpha_model=alpha_model,
            signals=signals,
            rebalance=rebalance_freq,
            long_only=True,
            cash_buffer_percentage=0.01,
            data_handler=strategy_data_handler,
            rebalance_weekday='FRI'
        )

        backtest.run()

        tearsheet = TearsheetStatistics(strategy_equity=backtest.get_equity_curve(), title='SMA')
        tearsheet.plot_results()
        results = tearsheet.get_results(backtest.get_equity_curve())
        tt = tearsheet.get_primary_results(results, "SMA")
        print(tt)
        profit_factor = tt['ProfitFactor']

        tearsheet = TearsheetStatistics(
            strategy_equity=backtest.get_equity_curve(),
            title='Cross Simple Moving Average vs SPY ETF'
        )

        # Dump results
        tearsheet.plot_results()
        
        return (profit_factor, short_window, long_window, tt)

    except Exception as e:
        # You might want to log this for debugging
        return (0, short_window, long_window, {"error": str(e)})

def run_optimization(start_dt, end_dt, strategy_universe, strategy_data_handler, rebalance_freq):
    short_range = range(5, 51)
    long_range = range(6, 201)  # since long_window must > short_window

    param_grid = [
        (short, long, start_dt, end_dt, strategy_universe, strategy_data_handler, rebalance_freq)
        for short in short_range
        for long in range(short + 1, 201)
    ]

    best_score = 0
    best_params = (0, 0)
    best_tt = None

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_combination, params) for params in param_grid]

        for future in as_completed(futures):
            profit_factor, short, long, tt = future.result()
            if profit_factor > best_score:
                best_score = profit_factor
                best_params = (short, long)
                best_tt = tt
                print('-------RESULTS-------', rebalance_freq)
                print("Best score", best_score)
                print("Best short", short)
                print("Best long", long)
                print(tt)

    print("Best parameters for the model")
    print("Short :", best_params[0])
    print("Long :", best_params[1])
    return best_params


if __name__ == "__main__":
    #start_hmm_dt = pd.Timestamp('1995-01-01 14:30:00', tz=pytz.UTC)
    #end_hmm_dt = pd.Timestamp('2024-01-01 00:00:00', tz=pytz.UTC)
    start_dt = pd.Timestamp('2015-01-01 14:30:00', tz=pytz.UTC)
    #burn_in_dt = pd.Timestamp('2004-11-22 14:30:00', tz=pytz.UTC)
    end_dt = pd.Timestamp('2025-02-28 23:59:00', tz=pytz.UTC)
    #Construct the symbol and asset necessary for the backtest
    strategy_symbols = ['SPY', 'AGG']
    strategy_assets = ['EQ:SPY', 'EQ:AGG']
    asset_dates = {asset: start_dt for asset in strategy_assets}
    strategy_universe = DynamicUniverse(asset_dates)
    #To avoid loading all CSV files in the directory, set the
    #data source to load only those provided symbols
    csv_dir = os.environ.get('QSTRADER_CSV_DATA_DIR', '.')
    strategy_data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols=strategy_symbols)
    strategy_data_handler = BacktestDataHandler(strategy_universe, data_sources=[strategy_data_source])

    #run_optimization(start_dt, end_dt, strategy_universe, strategy_data_handler, rebalance_freq="end_of_month")

    evaluate_combination([7, 8, start_dt, end_dt, strategy_universe, strategy_data_handler, "end_of_month"])