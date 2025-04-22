

from datetime import datetime
import json
import os
from examples.alpha_models.CrossCustomMovingAverage import CrossCustomMovingAverage
from examples.alpha_models.CrossWeightedMovingAverage import CrossWMovingAverage
from examples.alpha_models.CrossExponentialMovingAverage import CrossExponentialMovingAverage
from examples.alpha_models.SimpleCrossMovingAverage import SimpleCrossMovingAverage
from examples.factories.StategyFactory import StrategyFactory
from qstrader.alpha_model.fixed_signals import FixedSignalsAlphaModel
from qstrader.asset.equity import Equity
from qstrader.asset.universe.dynamic import DynamicUniverse
from qstrader.asset.universe.static import StaticUniverse
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.signals.current_price import CurrentPriceSignal
from qstrader.signals.ema import EMASignal
from qstrader.signals.signals_collection import SignalsCollection
from qstrader.signals.sma import SMASignal
from qstrader.signals.wma import WMASignal
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.backtest import BacktestTradingSession


class StrategyRunner:
    def __init__(
            self,
            start_dt,
            end_dt,
            symbols,
            name
        ):
        self.output_folder = "Results"
        self.name = name
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.symbols = symbols
        self.assets = [f"EQ:{symbol}" for symbol in self.symbols]

        # Create asset universe
        asset_dates = {asset: start_dt for asset in self.assets}
        self.strategy_universe = DynamicUniverse(asset_dates)

        # Create data source and handler
        self.csv_dir = os.environ.get('QSTRADER_CSV_DATA_DIR', '.')
        self.strategy_data_source = CSVDailyBarDataSource(self.csv_dir, Equity, csv_symbols=symbols)
        self.strategy_data_handler = BacktestDataHandler(self.strategy_universe, data_sources=[self.strategy_data_source])

    def get_strategy_factory(self, signals=None):
        return StrategyFactory(signals, self.strategy_universe, self.strategy_data_handler)
        
    def execute_sma(self, short_window, long_window, rebalance_freq="end_of_month", risk_model=None):
        plot_path, stat_path, desc_path = self._get_info_paths()

        sma = SMASignal(self.start_dt, self.strategy_universe, lookbacks=[short_window, long_window])
        signals = SignalsCollection({'sma': sma}, self.strategy_data_handler)

        alpha_model = SimpleCrossMovingAverage(
            signals,
            self.strategy_universe,
            self.strategy_data_handler,
            long_window,
            short_window
        )

        return self._execute_strategy(
            alpha_model,
            risk_model,
            signals,
            rebalance_freq,
            plot_path,
            stat_path,
            desc_path
        )
    
    def execute_ema(self, short_window, long_window, rebalance_freq="end_of_month", risk_model=None):
        plot_path, stat_path, desc_path = self._get_info_paths()
        ema = EMASignal(self.start_dt, self.strategy_universe, lookbacks=[short_window, long_window])
        signals = SignalsCollection({'ema': ema}, self.strategy_data_handler)

        alpha_model = CrossExponentialMovingAverage(
            signals,
            self.strategy_universe,
            self.strategy_data_handler,
            long_window,
            short_window
        )
        
        return self._execute_strategy(
            alpha_model,
            risk_model,
            signals,
            rebalance_freq,
            plot_path,
            stat_path,
            desc_path
        )

    def execute_wma(self, short_window, long_window, rebalance_freq="end_of_month", risk_model=None):
        plot_path, stat_path, desc_path = self._get_info_paths()
        wma = WMASignal(self.start_dt, self.strategy_universe, lookbacks=[short_window, long_window])
        signals = SignalsCollection({'wma': wma}, self.strategy_data_handler)

        alpha_model = CrossWMovingAverage(
            signals,
            self.strategy_universe,
            self.strategy_data_handler,
            long_window,
            short_window
        )
        
        return self._execute_strategy(
            alpha_model,
            risk_model,
            signals,
            rebalance_freq,
            plot_path,
            stat_path,
            desc_path
        )
    
    def execute_custom_cma(self, short_window, long_window, rebalance_freq="end_of_month", risk_model=None):
        # Custom cross moving average, which addresses ratios of prices moving averages
        plot_path, stat_path, desc_path = self._get_info_paths()
        wma = EMASignal(self.start_dt, self.strategy_universe, lookbacks=[short_window, long_window])
        # current securities price signal
        current_price = CurrentPriceSignal(self.start_dt, self.strategy_universe, lookbacks=[0,1,2])
        signals = SignalsCollection({'ema': wma, "cur_price": current_price}, self.strategy_data_handler)

        alpha_model = CrossCustomMovingAverage(
            signals,
            self.strategy_universe,
            self.strategy_data_handler,
            long_window,
            short_window
        )
        
        return self._execute_strategy(
            alpha_model,
            risk_model,
            signals,
            rebalance_freq,
            plot_path,
            stat_path,
            desc_path
        )

    def _execute_strategy(self, alpha_model, risk_model, signals, rebalance_freq, plot_path, stat_path, desc_path):
        backtest = BacktestTradingSession(
            self.start_dt,
            self.end_dt,
            universe=self.strategy_universe,
            alpha_model=alpha_model,
            risk_model=risk_model,
            signals=signals,
            rebalance=rebalance_freq,
            long_only=True,
            cash_buffer_percentage=0.01,
            data_handler=self.strategy_data_handler,
            rebalance_weekday='FRI'
        )
        backtest.run()
        benchmark_equity_curve = self.benchmark_spy()

        results = self._save_statistics(
            backtest.get_equity_curve(),
            benchmark_equity_curve,
            self._get_title(alpha_model, risk_model),
            plot_path,
            stat_path,
            rebalance_freq
        )

        description = risk_model.get_description() if risk_model is not None else "Risk model is None"
        self._save_description(desc_path, alpha_model.get_description(), description)
        return results
    
    def _get_title(self, alpha_model, risk_model):
        if risk_model is None:
            title = alpha_model.get_description()['name'] + " vs " + "SPY"
        else:
            title = "HMM " + alpha_model.get_description()['name'] + " vs " + "SPY"
        return title
    
    def benchmark_spy(self):
        #Construct a benchmark Alpha Model that provides
        #100% static allocation to the SPY ETF, with no rebalance
        #Construct benchmark assets (buy & hold SPY)
        benchmark_assets = ['EQ:SPY']
        benchmark_universe = StaticUniverse(benchmark_assets)
        benchmark_alpha_model = FixedSignalsAlphaModel({'EQ:SPY': 1.0})
        benchmark_backtest = BacktestTradingSession(
            self.start_dt,
            self.end_dt,
            benchmark_universe,
            benchmark_alpha_model,
            rebalance='buy_and_hold',
            long_only=True,
            cash_buffer_percentage=0.1,
            data_handler=self.strategy_data_handler
        )
        benchmark_backtest.run()
        return benchmark_backtest.get_equity_curve()
    

    def _get_info_paths(self):
        run_number = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        subfolder_path = os.path.join(self.output_folder, self.name, run_number)
        os.makedirs(subfolder_path)
        plot_path = os.path.join(subfolder_path, "plot.png")
        stat_path = os.path.join(subfolder_path, "output.json")
        desc_path = os.path.join(subfolder_path, "desc.json")
        return plot_path, stat_path, desc_path
    
    def _save_statistics(self, backtest_curve, benchmark_curve, title, plot_path, stat_path, rebalance_freq):
        tearsheet = TearsheetStatistics(
            strategy_equity=backtest_curve,
            benchmark_equity=benchmark_curve,
            title=title
        )

        #tearsheet.plot_results(filename=plot_path)

        # Get backtest results
        backtest_result_stats = tearsheet.get_results(backtest_curve)
        backtest_stats = tearsheet.get_primary_results(backtest_result_stats, title)

        # Save necessary metada
        backtest_stats['rebalance_freq'] = rebalance_freq
        backtest_stats['start_dt'] = str(self.start_dt)
        backtest_stats['end_dt'] = str(self.end_dt)

        # Get benchmark results
        if benchmark_curve is not None:
            bench_stats = tearsheet.get_results(benchmark_curve)
            bench_stats = tearsheet.get_primary_results(bench_stats, title)

        # Statistics
        stats = {
            "Strategy": backtest_stats,
            "Benchmark": bench_stats
        }

        # Serialize
        with open(stat_path, "w") as json_file:
            json.dump(stats, json_file, indent=4)

        return backtest_stats

    def _save_description(self, desc_path, alpha_model_desc, risk_model_desc):
        description = {
            "AlphaModel": alpha_model_desc,
            "RiskModel": risk_model_desc
        }
    
        with open(desc_path, "w") as json_file:
            json.dump(description, json_file, indent=4)
