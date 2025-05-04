

from datetime import datetime
import json
import os
import matplotlib.pyplot as plt

import pandas as pd
from examples.alpha_models.RsiModel import RSIBasedAlphaModel
from examples.alpha_models.CrossCustomMovingAverage import CrossCustomMovingAverage
from examples.alpha_models.CrossWeightedMovingAverage import CrossWMovingAverage
from examples.alpha_models.CrossExponentialMovingAverage import CrossExponentialMovingAverage
from examples.alpha_models.KamaModel import KamaModel
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
from qstrader.signals.kama import KaufmanAdaptiveMASignal
from qstrader.signals.rsi import RSISignal
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
            name,
            show_plot = False,
            run_benchmark = False
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

        # Params
        self.show_plot = show_plot
        self.run_benchmark = run_benchmark

    def get_strategy_factory(self, signals=None):
        return StrategyFactory(signals, self.strategy_universe, self.strategy_data_handler)
        
    def execute_sma(self, short_window=10, long_window=30, rebalance_freq="end_of_month", risk_model=None):
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
    
    def execute_ema(self, short_window=10, long_window=30, rebalance_freq="end_of_month", risk_model=None):
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

    def execute_wma(self, short_window=10, long_window=30, rebalance_freq="end_of_month", risk_model=None):
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
    
    def get_rewards_by_epoch(self, performance_stats):
        """
        Get the latest reward value for each epoch from the performance_stats DataFrame.

        Parameters
        ----------
        performance_stats : pd.DataFrame
            A DataFrame with a MultiIndex (Date, Epoch) and a 'Reward' column.

        Returns
        -------
        pd.Series
            A Series with the latest reward for each epoch.
        """
        # Group by the 'Epoch' level of the MultiIndex and get the last reward for each epoch
        rewards_by_epoch = performance_stats.groupby(level='Epoch')['Reward'].last()

        # Print the rewards for each epoch
        print("Rewards by Epoch:")
        print(rewards_by_epoch)


    def plot_rewards_by_epoch(self, performance_stats):
        """
        Get the latest reward for each epoch from the performance_stats DataFrame
        and plot it as a line chart.

        Parameters
        ----------
        performance_stats : pd.DataFrame
            A DataFrame with a MultiIndex (Date, Epoch) and a 'Reward' column.

        Returns
        -------
        pd.Series
            A Series with the latest reward for each epoch.
        """
        # Group by the 'Epoch' level of the MultiIndex and get the last reward for each epoch
        rewards_by_epoch = performance_stats.groupby(level='Epoch')['Reward'].last()

        # Plot the rewards as a line chart
        plt.figure(figsize=(10, 6))
        plt.plot(rewards_by_epoch.index, rewards_by_epoch.values, marker='o', linestyle='-', color='blue', label='Reward')
        plt.title("Latest Reward by Epoch", fontsize=16)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Reward", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rewards_by_epoch.index, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def execute_QLearning(self, epochs=200, learning_rate=0.001, discount_factor=0.9, exploration_rate=1.0, rebalance_freq="end_of_month", risk_model=None):
        """
        Execute a Q-Learning-based alpha model strategy.
        """

        # Define signals
        short_sma = 8
        long_sma = 20

        short_wma = 12
        long_wma = 32

        long_ema = 10
        short_ema = 40

        kama_period = 365
        rsi_period_long = 30
        rsi_period_short = 13
        
        kama_period = 365
        sma = SMASignal(self.start_dt, self.strategy_universe, lookbacks=[short_sma, long_sma])
        wma = WMASignal(self.start_dt, self.strategy_universe, lookbacks=[short_wma, long_wma])
        ema = EMASignal(self.start_dt, self.strategy_universe, lookbacks=[short_ema, long_ema])
        rsi = RSISignal(self.start_dt, self.strategy_universe, lookbacks=[rsi_period_long, rsi_period_short])
        current_price = CurrentPriceSignal(self.start_dt, self.strategy_universe, lookbacks=[0,1,2])
        kama = KaufmanAdaptiveMASignal(self.start_dt, self.strategy_universe, lookbacks=[kama_period], fast_period=38, slow_period=39, er_period=100)
        signals = SignalsCollection(
        {
            'sma': sma,
            'wma': wma, 
            'ema': ema,
            'rsi': rsi,
            'cur_price': current_price,
            'kama': kama
        },
            self.strategy_data_handler)
        
        benchmark_curve = self.benchmark_spy(self.start_dt, self.end_dt)
        # Initialize the Q-Learning Alpha Model
        performance_stats = pd.DataFrame(columns=[
            'Action', 'Returns', 'Sharpe', 'Profit Factor', 'Benchmark Returns', 'Reward'
        ])
        performance_stats.index = pd.MultiIndex.from_tuples([], names=["Date", "Epoch"])
        
        from examples.alpha_models.QLearningAlphaModel import QLearningAlphaModel
        alpha_model = QLearningAlphaModel(
            signals,
            self.strategy_universe,
            self.strategy_data_handler,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate,
            benchmark_curve=benchmark_curve,
            performance_stats=performance_stats
        )

        run_number = self._get_run_number()
        # Execute the strategy
        for e in range(epochs):
            plot_path, stat_path, desc_path = self._get_info_paths(prefix=f"Train_{e}")

            print("Starting epoch", e)
            # Train the Q-Learning model
            alpha_model.initialize(e)
            alpha_model.set_eval_mode(False)
            results = self._execute_strategy(
                alpha_model,
                risk_model,
                signals,
                rebalance_freq,
                plot_path,
                stat_path,
                desc_path
            )
            print("End epoch", e)
            print("Profit factor and returns", results['ProfitFactor'], results['TotalReturns'])
            self.get_rewards_by_epoch(performance_stats)

            plot_path, stat_path, desc_path = self._get_info_paths(prefix=f"Eval_{e}")
            # Evalute in-sample performance
            self.show_plot = False
            self.run_benchmark = True
            alpha_model.set_eval_mode(True)
            results = self._execute_strategy(
                alpha_model,
                risk_model,
                signals,
                rebalance_freq,
                plot_path,
                stat_path,
                desc_path
            )
            print(f"-------[IN SAMPLE RESULT]-------Epoch {e}/{epochs}-------")
            print(results)
            performance_stats.to_csv("performance_stats_"+ run_number +".csv")
            print("--------------------------------")
        self.plot_rewards_by_epoch(performance_stats)


    def execute_custom_cma(self, short_window=10, long_window=30, rebalance_freq="end_of_month", risk_model=None):
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
    
    def execute_kama(self, fast_period=2, slow_period=30, er_period=10, kama_period=365, rebalance_freq="end_of_month", risk_model=None):
        plot_path, stat_path, desc_path = self._get_info_paths()
        kama = kama = KaufmanAdaptiveMASignal(self.start_dt, self.strategy_universe, lookbacks=[kama_period], fast_period=fast_period, slow_period=slow_period, er_period=er_period)
        current_price = CurrentPriceSignal(self.start_dt, self.strategy_universe, lookbacks=[0,1,2])
        signals = SignalsCollection({'kama': kama, 'current_price': current_price}, self.strategy_data_handler)

        alpha_model = KamaModel(
            signals,
            self.strategy_universe,
            self.strategy_data_handler,
            kama_period
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
    
    def execute_rsi(self, rsi_period=14, rsi_upper=70, rsi_lower=30, rebalance_freq="end_of_month", risk_model=None):
        plot_path, stat_path, desc_path = self._get_info_paths()
        rsi = RSISignal(self.start_dt, self.strategy_universe, lookbacks=[rsi_period])
        signals = SignalsCollection({'rsi': rsi}, self.strategy_data_handler)

        alpha_model = RSIBasedAlphaModel(
            signals,
            self.strategy_universe,
            self.strategy_data_handler,
            rsi_period = rsi_period,
            rsi_upper=rsi_upper,
            rsi_lower=rsi_lower
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
    
    def execute_benchmark(self, rebalance_freq="end_of_month", risk_model=None):
        benchmark_curve = self.benchmark_spy(self.start_dt, self.end_dt)
        title="Benchmark Long SPY"

        tearsheet = TearsheetStatistics(
            strategy_equity=None,
            benchmark_equity=benchmark_curve,
            title=title
        )

        bench_res = tearsheet.get_results(benchmark_curve)
        bench_stats = tearsheet.get_primary_results(bench_res, title)
        # Save necessary metada
        bench_stats['rebalance_freq'] = rebalance_freq
        bench_stats['start_dt'] = str(self.start_dt)
        bench_stats['end_dt'] = str(self.end_dt)
        return bench_stats


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

        benchmark_equity_curve = None
        if self.run_benchmark:
            benchmark_equity_curve = self.benchmark_spy(self.start_dt, self.end_dt)

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
    
    def benchmark_spy(self, start_dt, end_dt):
        #Construct a benchmark Alpha Model that provides
        #100% static allocation to the SPY ETF, with no rebalance
        #Construct benchmark assets (buy & hold SPY
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
            cash_buffer_percentage=0.01,
            data_handler=self.strategy_data_handler,
        )
        benchmark_backtest.run()
        return benchmark_backtest.get_equity_curve()
    

    def _get_run_number(self):
        now = datetime.now()
        run_number = now.strftime("%Y-%m-%d-%H:%M:%S") + f".{int(now.microsecond / 1000):03d}"
        return run_number
    
    def _get_info_paths(self, prefix=None):
        run_number = self._get_run_number()
        if prefix is not None:
            run_number = str(prefix) + "_" + run_number
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


        tearsheet.plot_results(filename=plot_path, show_plot=self.show_plot)

        backtest_stats = None
        # Get backtest results
        if backtest_curve is not None:
            backtest_result_stats = tearsheet.get_results(backtest_curve)
            backtest_stats = tearsheet.get_primary_results(backtest_result_stats, title)
                    # Save necessary metada
            backtest_stats['rebalance_freq'] = rebalance_freq
            backtest_stats['start_dt'] = str(self.start_dt)
            backtest_stats['end_dt'] = str(self.end_dt)


        # Get benchmark results
        bench_stats = None
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
