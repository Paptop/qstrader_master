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
from examples.alpha_models.RsiModel import RSIBasedAlphaModel
from examples.alpha_models.KamaModel import KamaModel
from qstrader.signals.rsi import RSISignal
from qstrader.signals.kama import KaufmanAdaptiveMASignal
from examples.alpha_models.CrossExponentialMovingAverage import CrossExponentialMovingAverage
from examples.alpha_models.CrossWeightedMovingAverage import CrossWMovingAverage
from examples.plots.plot import plot_sp500_with_regime
from examples.risk_filters.HMM import train_hmm
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


if __name__ == "__main__":
    strategy_name = "hmmFilter_short_long_performance"
    output_folder = "Results"
    run_number = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    subfolder_path = os.path.join(output_folder, strategy_name, run_number)
    os.makedirs(subfolder_path)
    plot_path = os.path.join(subfolder_path, "plot.png")
    stat_path = os.path.join(subfolder_path, "output.json")
    desc_path = os.path.join(subfolder_path, "desc.json")

    # Out of sample HMM training
    start_hmm_dt = pd.Timestamp('1995-01-01 14:30:00', tz=pytz.UTC)
    end_hmm_dt = pd.Timestamp('2015-01-01 00:00:00', tz=pytz.UTC)
    start_dt = pd.Timestamp('2015-01-01 14:30:00', tz=pytz.UTC)
    #burn_in_dt = pd.Timestamp('2004-11-22 14:30:00', tz=pytz.UTC)
    end_dt = pd.Timestamp('2025-02-28 23:59:00', tz=pytz.UTC)

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

    # Generate the signals (in this case holding-period return based
    # momentum) used in the top-N momentum alpha model
    short = 50
    long = 200
    rsi_period = 13
    kama_period = 365
    sma = SMASignal(start_dt, strategy_universe, lookbacks=[short, long])
    wma = WMASignal(start_dt, strategy_universe, lookbacks=[short, long])
    ema = EMASignal(start_dt, strategy_universe, lookbacks=[short, long])
    rsi = RSISignal(start_dt, strategy_universe, lookbacks=[rsi_period])
    kama = KaufmanAdaptiveMASignal(start_dt, strategy_universe, lookbacks=[kama_period], fast_period=20, slow_period=300, er_period=100)
    signals = SignalsCollection({'sma': sma, 'wma': wma, 'ema': ema, 'kama': kama, 'rsi': rsi}, strategy_data_handler)

    strategy_cross_moving_average_alpha_model = CrossCustomMovingAverage(
        signals,
        strategy_universe,
        strategy_data_handler,
        long,
        short
    )

    strategy_cross_wmoving_average_alpha_model = CrossWMovingAverage(
        signals,
        strategy_universe,
        strategy_data_handler,
        long,
        short
    )

    strategy_cross_exp_moving_average_alpha_model = CrossExponentialMovingAverage(
        signals,
        strategy_universe,
        strategy_data_handler,
        long,
        short
    )

    strategy_rsi_based_alpha_model = RSIBasedAlphaModel(
        signals,
        strategy_universe,
        strategy_data_handler,
        rsi_period,
        rsi_upper=70,
        rsi_lower=30
    )
    
    prices_for_kama = strategy_data_source.get_assets_historical_closes(start_hmm_dt, end_dt, ['EQ:SPY'])
    prices_for_kama.index = prices_for_kama.index.date

    strategy_kama_alpha_model = KamaModel(
        signals,
        strategy_universe,
        strategy_data_handler,
        kama_period,
        prices_for_kama
    )

    strategy_alpha_model = strategy_cross_moving_average_alpha_model

    # Use an example Risk Manager
    #risk_manager = ExampleRiskManager()
    # Use regime detection HMM risk manager
    prices = strategy_data_source.get_assets_historical_closes(start_hmm_dt, end_dt, ['EQ:SPY']).pct_change().dropna()
    prices.index = prices.index.date
    prices['regime'] = 0
    #hmm_model = pickle.load(open('./examples/hmm_model_spy.pkl', "rb"))

    # Create the Gaussian Hidden markov Model and fit it
    # to the SPY returns data, outputting a score
    hmm_model = train_hmm(start_hmm_dt, end_hmm_dt, prices)
    #plot_in_sample_hidden_states(hmm_model, train_data, rets)

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
    strategy_risk_model = RiskHMMFilter(
        hmm_model=hmm_model,
        universe=strategy_universe,
        adj_price_returns=prices,
        alpha_model=strategy_alpha_model,
        params=params,
        isFuture=False
    )

    # Construct the transaction cost modelling - fees/slippage
    # 0.1% transaction fee
    # 0.5% tax on each transaction
    fee_model = PercentFeeModel(commission_pct=0.01 / 100.0, tax_pct=0.05 / 100.0)
    
    strategy_backtest = BacktestTradingSession(
        start_dt,
        end_dt,
        universe=strategy_universe,
        alpha_model=strategy_alpha_model,
        #risk_model=strategy_risk_model,
        signals=signals,
        rebalance='end_of_month',
        long_only=True,
        cash_buffer_percentage=0.01,
        data_handler=strategy_data_handler,
        #burn_in_dt=burn_in_dt,
        rebalance_weekday='FRI',
        #fee_model=fee_model
    )
    strategy_backtest.run()
    #plot_regime(prices)


    # Construct a benchmark Alpha Model that provides
    # 100% static allocation to the SPY ETF, with no rebalance
    # Construct benchmark assets (buy & hold SPY)
    benchmark_assets = ['EQ:SPY']
    benchmark_universe = StaticUniverse(benchmark_assets)
    benchmark_alpha_model = FixedSignalsAlphaModel({'EQ:SPY': 1.0})

    benchmark_backtest = BacktestTradingSession(
        start_dt,
        end_dt,
        benchmark_universe,
        benchmark_alpha_model,
        rebalance='buy_and_hold',
        long_only=True,
        cash_buffer_percentage=0.1,
        data_handler=strategy_data_handler
    )
    benchmark_backtest.run()

    sp500 = strategy_data_source.get_assets_historical_closes(end_hmm_dt, end_dt, ['EQ:SPY'])
    sp500.index = sp500.index.date
    sp500['regime'] = prices['regime']
    plot_sp500_with_regime(sp500)

    # Performance Output
    tearsheet = TearsheetStatistics(
        strategy_equity=strategy_backtest.get_equity_curve(),
        benchmark_equity=benchmark_backtest.get_equity_curve(),
        title='HMM WMA vs SPY ETF'
    )

    # Dump results
    tearsheet.plot_results(filename=plot_path)
    results = tearsheet.get_results(strategy_backtest.get_equity_curve())

    tt = tearsheet.get_primary_results(results, strategy_name)
    with open(stat_path, "w") as json_file:
        json.dump(tt, json_file, indent=4)
    
    description = {
        "AlphaModel": strategy_alpha_model.get_description(),
        "RiskModel": strategy_risk_model.get_description()
    }
    
    with open(desc_path, "w") as json_file:
        json.dump(description, json_file, indent=4)
