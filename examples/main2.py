import os
import sys

import numpy as np
import pandas as pd
import pytz
import json
from datetime import datetime

# Get the absolute path to the cloned library
lib_path = os.path.abspath("./")  # Adjust the path as needed

# Add to sys.path if not already present
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

# Local libs
from examples.alpha_models.SimpleCrossMovingAverage import SimpleCrossMovingAverage
from examples.factories.StategyFactory import StrategyFactory
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

def train_and_optimize_sma(start_dt, end_dt, strategy_universe, strategy_data_handler, rebalance_freq):
    best_score = 0
    opt_short = 0
    opt_long = 0

    short_window = 7
    long_window = 8

    sma = SMASignal(start_dt, strategy_universe, lookbacks=[short_window, long_window])
    signals = SignalsCollection({'sma': sma}, strategy_data_handler) 
    # Construct the transaction cost modelling - fees/slippage
    # 0.1% transaction fee
    # 0.5% tax on each transaction
    #fee_model = PercentFeeModel(commission_pct=0.01 / 100.0, tax_pct=0.05 / 100.

    strategy_factory = StrategyFactory(signals, strategy_universe, strategy_data_handler)
    strategy_backtest = BacktestTradingSession(
        start_dt,
        end_dt,
        universe=strategy_universe,
        alpha_model=strategy_factory.create_strategy(
            strategy_type="simple_cross_moving_average",
            long=long_window,
            short=short_window
        ),
        #risk_model=strategy_risk_model,
        signals=signals,
        rebalancer=rebalance_freq,
        long_only=True,
        cash_buffer_percentage=0.01,
        data_handler=strategy_data_handler,
        #burn_in_dt=burn_in_dt,
        rebalance_weekday='FRI',
        #fee_model=fee_model
    )
    strategy_backtest.run()
    # Performance Output
    tearsheet = TearsheetStatistics(
        strategy_equity=strategy_backtest.get_equity_curve(),
        title='SMA'
    )
    results = tearsheet.get_results(strategy_backtest.get_equity_curve())
    tt = tearsheet.get_primary_results(results, "SMA")
    profitFactor = tt['ProfitFactor']
    if profitFactor > best_score:
        best_score = profitFactor
        opt_short = short_window
        opt_long = long_window
        print('-------RESULTS-------', rebalance_freq)
        print("Best score", best_score)
        print("Best short", opt_short)
        print("Best long", opt_long)
        print(tt)

    print("Best parameters for the model")
    print("Short :", opt_short)
    print("Long :", opt_long)
    return opt_short, opt_long
    
def primary_logic():
    strategy_name = "ResultsGathering1.0"
    output_folder = "Results"
    run_number = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    subfolder_path = os.path.join(output_folder, strategy_name, run_number)
    os.makedirs(subfolder_path)
    plot_path = os.path.join(subfolder_path, "plot.png")
    stat_path = os.path.join(subfolder_path, "output.json")
    desc_path = os.path.join(subfolder_path, "desc.json")
    data_loader = SimpleDataLoader()

    # Out of sample HMM training
    #start_hmm_dt = pd.Timestamp('1995-01-01 14:30:00', tz=pytz.UTC)
    #end_hmm_dt = pd.Timestamp('2015-01-01 00:00:00', tz=pytz.UTC)
    start_dt = pd.Timestamp('2015-01-01 14:30:00', tz=pytz.UTC)
    end_dt = pd.Timestamp('2025-02-28 23:59:00', tz=pytz.UTC)
    #burn_in_dt = pd.Timestamp('2004-11-22 14:30:00', tz=pytz.UTC)

    # Construct the symbol and asset necessary for the backtest
    strategy_symbols = ['SPY', 'AGG']
    strategy_assets = ['EQ:SPY', 'EQ:AGG']
    asset_dates = {asset: start_dt for asset in strategy_assets}
    strategy_universe = DynamicUniverse(asset_dates)

    # To avoid loading all CSV files in the directory, set the
    # data source to load only those provided symbols
    csv_dir = os.environ.get('QSTRADER_CSV_DATA_DIR', '.')
    strategy_data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols=strategy_symbols)
    strategy_data_handler = BacktestDataHandler(strategy_universe, data_sources=[strategy_data_source])

    short_window = 7
    long_window = 8
    rebalance_freq = "end_of_month"
    sma = SMASignal(start_dt, strategy_universe, lookbacks=[long_window, short_window])
    signals = SignalsCollection({'sma': sma}, strategy_data_handler) 
    # Construct the transaction cost modelling - fees/slippage
    # 0.1% transaction fee
    # 0.5% tax on each transaction
    #fee_model = PercentFeeModel(commission_pct=0.01 / 100.0, tax_pct=0.05 / 100.

    rsi_period = 13
    kama_period = 365
    #sma = SMASignal(start_dt, strategy_universe, lookbacks=[short, long])
    #wma = WMASignal(start_dt, strategy_universe, lookbacks=[short, long])
    #ema = EMASignal(start_dt, strategy_universe, lookbacks=[short, long])
    #rsi = RSISignal(start_dt, strategy_universe, lookbacks=[rsi_period])
    #current_price = CurrentPriceSignal(start_dt, strategy_universe, lookbacks=[0,1,2])
    #kama = KaufmanAdaptiveMASignal(start_dt, strategy_universe, lookbacks=[kama_period], fast_period=38, slow_period=39, er_period=100)

    
    # strategy_cross_moving_average_alpha_model = CrossMovingAverage(
    #     signals,
    #     strategy_universe,
    #     strategy_data_handler,
    #     long,
    #     short
    # )

    # strategy_cross_wmoving_average_alpha_model = CrossWMovingAverage(
    #     signals,
    #     strategy_universe,
    #     strategy_data_handler,
    #     long,
    #     short
    # )

    # strategy_cross_exp_moving_average_alpha_model = CrossExponentialMovingAverage(
    #     signals,
    #     strategy_universe,
    #     strategy_data_handler,
    #     long,
    #     short
    # )

    # strategy_rsi_based_alpha_model = RSIBasedAlphaModel(
    #     signals,
    #     strategy_universe,
    #     strategy_data_handler,
    #     rsi_period,
    #     rsi_upper=70,
    #     rsi_lower=30
    # )
    
    # prices_for_kama = strategy_data_source.get_assets_historical_closes(start_hmm_dt, end_dt, ['EQ:SPY'])
    # prices_for_kama.index = prices_for_kama.index.date

    # strategy_kama_alpha_model = KamaModel(
    #     signals,
    #     strategy_universe,
    #     strategy_data_handler,
    #     kama_period,
    #     prices_for_kama
    # )

    strategy_factory = StrategyFactory(signals, strategy_universe, strategy_data_handler)
    strategy_backtest = BacktestTradingSession(
        start_dt,
        end_dt,
        universe=strategy_universe,
        alpha_model= strategy_factory.create_strategy(
            strategy_type="simple_cross_moving_average",
            long=long_window,
            short=short_window
        ),
        #risk_model=strategy_risk_model,
        signals=signals,
        rebalance='end_of_month',
        long_only=True,
        cash_buffer_percentage=0.01,
        data_handler=strategy_data_handler,
        #burn_in_dt=burn_in_dt,
        rebalance_weekday='FRI',
        #fee_model=fee_model_monthly
    )
    strategy_backtest.run()

    tearsheet = TearsheetStatistics(
        strategy_equity=strategy_backtest.get_equity_curve(),
        #benchmark_equity=benchmark_backtest.get_equity_curve(),
        title='Cross Simple Moving Average vs SPY ETF'
    )

    # Dump results
    tearsheet.plot_results(filename=plot_path)
    results = tearsheet.get_results(strategy_backtest.get_equity_curve())


    #strategy_cross_wmoving_average_alpha_model

    # Use an example Risk Manager
    #risk_manager = ExampleRiskManager()
    # Use regime detection HMM risk manager
    #plot_in_sample_hidden_states(hmm_model, train_data, rets)

    # merged_data = data_loader.get_data_frame(start_hmm_dt, end_dt)

    # params = {
    #     "smooth": False,
    #     "easing": False,
    #     "lag": False,
    #     "current_day_only": True,
    #     "random": False,
    #     "smooth_days": 10,  # Default 19
    #     "lag_in_days": 0,  # Default 0
    #     "window_size": 365  # Default 365
    # }
    # strategy_risk_model = MultiRiskHMMFilter(
    #     universe=strategy_universe,
    #     alpha_model=strategy_alpha_model,
    #     data=merged_data,
    #     features=
    #     {
    #         'Adj_Close_SPY_change': { 'w': 1.0, 'sort_by_returns': True, 'flip_values': False },
    #         #'Volume_SPY_change':  { 'w': 0.1, 'sort_by_returns': True, 'flip_values': True },
    #         #'Adj_Close_AAPL_change':  { 'w': 1.0, 'sort_by_returns': True }
    #         #'Volume_AAPL_change':  { 'w': 1.0, 'sort_by_returns': False }
    #         'Adj_Close_VIX_change':  { 'w': 1.0, 'sort_by_returns': False, 'flip_values': False },
    #         #'10Y_Treasury':  { 'w': 0.01, 'sort_by_returns': True, 'flip_values': False },
    #         #'2Y_Treasury':  { 'w': 0.01, 'sort_by_returns': True, 'flip_values': False },
    #         #'CPI': { 'w': 0.5, 'sort_by_returns': False, 'flip_values': True }
    #     },
    #     params=params,
    #     start_hmm_dt=start_hmm_dt,
    #     end_hmm_dt=end_hmm_dt
    # )

    # Construct the transaction cost modelling - fees/slippage
    # 0.1% transaction fee
    # 0.5% tax on each transaction
    #fee_model_daily = PercentFeeModel( commission_pct=0.005 / 100.0, tax_pct=0.001 / 100.0)
    #fee_model_monthly = PercentFeeModel(commission_pct=0.01 / 100.0, tax_pct=0.25 / 100.0)


    #plot_regime(prices)


    # Construct a benchmark Alpha Model that provides
    # 100% static allocation to the SPY ETF, with no rebalance
    # Construct benchmark assets (buy & hold SPY)
    # benchmark_assets = ['EQ:SPY']
    # benchmark_universe = StaticUniverse(benchmark_assets)
    # benchmark_alpha_model = FixedSignalsAlphaModel({'EQ:SPY': 1.0})

    # benchmark_backtest = BacktestTradingSession(
    #     start_dt,
    #     end_dt,
    #     benchmark_universe,
    #     benchmark_alpha_model,
    #     rebalance='buy_and_hold',
    #     long_only=True,
    #     cash_buffer_percentage=0.1,
    #     data_handler=strategy_data_handler,
    #     fee_model=fee_model_monthly
    # )
    # benchmark_backtest.run()

    #sp500 = merged_data[['Adj_Close_SPY', 'pregime_label']] #strategy_data_source.get_assets_historical_closes(end_hmm_dt, end_dt, ['EQ:SPY'])
    #sp500.index = sp500.index.date
    #plot_sp500_with_regime(sp500, 'Adj_Close_SPY')

    # Performance Output
    tearsheet = TearsheetStatistics(
        strategy_equity=strategy_backtest.get_equity_curve(),
        #benchmark_equity=benchmark_backtest.get_equity_curve(),
        title='Cross Simple Moving Average vs SPY ETF'
    )

    # Dump results
    tearsheet.plot_results(filename=plot_path)
    results = tearsheet.get_results(strategy_backtest.get_equity_curve())

    tt = tearsheet.get_primary_results(results, strategy_name)
    print(tt)
    with open(stat_path, "w") as json_file:
        json.dump(tt, json_file, indent=4)
    
    description = {
        "AlphaModel": None, #strategy_alpha_model.get_description(),
        "RiskModel": None, #strategy_risk_model.get_description()
    }
    
    with open(desc_path, "w") as json_file:
        json.dump(description, json_file, indent=4)


if __name__ == "__main__":
    #Out of sample HMM training
    # start_hmm_dt = pd.Timestamp('1995-01-01 14:30:00', tz=pytz.UTC)
    # end_hmm_dt = pd.Timestamp('2024-01-01 00:00:00', tz=pytz.UTC)
    # start_dt = pd.Timestamp('2015-01-01 14:30:00', tz=pytz.UTC)
    # burn_in_dt = pd.Timestamp('2004-11-22 14:30:00', tz=pytz.UTC)
    # end_dt = pd.Timestamp('2025-02-28 23:59:00', tz=pytz.UTC)
    #Construct the symbol and asset necessary for the backtest
    # strategy_symbols = ['SPY', 'AGG']
    # strategy_assets = ['EQ:SPY', 'EQ:AGG']
    # asset_dates = {asset: start_dt for asset in strategy_assets}
    # strategy_universe = DynamicUniverse(asset_dates)
    #To avoid loading all CSV files in the directory, set the
    #data source to load only those provided symbols
    # csv_dir = os.environ.get('QSTRADER_CSV_DATA_DIR', '.')
    # strategy_data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols=strategy_symbols)
    # strategy_data_handler = BacktestDataHandler(strategy_universe, data_sources=[strategy_data_source])#
    #train_and_optimize_sma(start_dt, end_dt, strategy_universe, strategy_data_handler, "end_of_month")
    #train_and_optimize_sma(start_dt, end_dt, strategy_universe, strategy_data_handler, "weekly")
    #train_and_optimize_sma(start_dt, end_dt, strategy_universe, strategy_data_handler, "daily")
    primary_logic()