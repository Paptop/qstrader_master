import os
import sys

from matplotlib import pyplot as plt
import seaborn as sns
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
    start_hmm_dt = start_dt - pd.DateOffset(days=params['window_size']) #start_dt #pd.Timestamp('1995-01-01 14:30:00', tz=pytz.UTC)
    end_hmm_dt = start_dt - pd.DateOffset(days=7)
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

def execute_strategies_and_create_dataframe(strategy_runner, risk_model, rebalance_freq):
    # Execute strategies
    sma = strategy_runner.execute_sma(rebalance_freq=rebalance_freq, risk_model=risk_model)
    ema = strategy_runner.execute_ema(rebalance_freq=rebalance_freq, risk_model=risk_model)
    wma = strategy_runner.execute_wma(rebalance_freq=rebalance_freq, risk_model=risk_model)
    cma = strategy_runner.execute_custom_cma(rebalance_freq=rebalance_freq, risk_model=risk_model)
    rsi = strategy_runner.execute_rsi(rebalance_freq=rebalance_freq, risk_model=risk_model)
    kama = strategy_runner.execute_kama(rebalance_freq=rebalance_freq, risk_model=risk_model)

    # Optional benchmark values
    benchmark = strategy_runner.execute_benchmark(rebalance_freq=rebalance_freq, risk_model=risk_model)

    # Collect results
    results = []
    for res in [benchmark, sma, ema, wma, cma, rsi, kama]:
        results.append({
            'Strategy': res['Strategy'],
            'ProfitFactor': round(float(res['ProfitFactor']), 2),
            'SharpeRatio': round(float(res['Sharpe']), 2),
            'SortinoRatio': round(float(res['Sortino']), 2),
            'TotalReturns': round(float(res['TotalReturns']), 2)
        })

    # Create the dataframe
    df = pd.DataFrame(results)
    # df = pd.concat([
    #     df,
    #     pd.DataFrame([{
    #         'Strategy': 'Rebalance',
    #         'ProfitFactor': 'Monthly',
    #         'SharpeRatio': 'Monthly',
    #         'SortinoRatio': 'Monthly',
    #         'TotalReturns': 'Monthly'
    #     }, {
    #         'Strategy': 'RiskFilter',
    #         'ProfitFactor': 'On',
    #         'SharpeRatio': 'On',
    #         'SortinoRatio': 'On',
    #         'TotalReturns': 'On'
    #     }])
    # ], ignore_index=True)

    df.columns = ['Strategy', 'ProfitFactor', 'SharpeRatio', 'SortinoRatio', 'TotalReturns']
    return df


def compare_and_plot(start_dt, end_dt, symbols, name, rebalance_freq="end_of_month"):
    strategy_runner = StrategyRunner(start_dt, end_dt, symbols, name, False, False)
    risk_model = create_risk_model(strategy_runner.get_strategy_factory(), start_dt, end_dt)

    df_without_risk = execute_strategies_and_create_dataframe(strategy_runner, None, rebalance_freq)
    df_risk = execute_strategies_and_create_dataframe(strategy_runner, risk_model, rebalance_freq)

    # Add identifiers before merging
    df_without_risk['risk_applied'] = False
    df_risk['risk_applied'] = True

    print("---Without Risk---")
    print(df_without_risk)

    print("---With Risk filter---")
    print(df_risk)

    # Concatenate both DataFrames
    merged_df = pd.concat([df_risk, df_without_risk], ignore_index=True)
    merged_df['start_dt'] = start_dt
    merged_df['end_dt'] = end_dt
    merged_df['rebalance_freq'] = rebalance_freq
    
    return merged_df
    # Define metrics for which we want deltas
    #metrics = ['ProfitFactor', 'SharpeRatio', 'SortinoRatio', 'TotalReturns']
    # Create dictionary to hold delta values
    # delta_data = {
    #     'Strategy': df_risk['Strategy']
    # }
    # Calculate deltas
    # for metric in metrics:
    #     base = df_without_risk[metric].replace(0, pd.NA)
    #     delta = ((df_risk[metric] - df_without_risk[metric]) / base) * 100
    #     delta_data[f'{metric}_Delta'] = delta

    # # Create the delta DataFrame
    # df_deltas = pd.DataFrame(delta_data)

    # # Optional: round for presentation
    # df_deltas = df_deltas.round(2)

    # # Display the dataframe
    # print(df_deltas)

    # # Make sure 'Strategy' is index
    # df_heat = df_deltas.set_index('Strategy')

    # # Optional: remove the Benchmark row to focus on actual strategies
    # df_heat = df_heat.drop(index='Benchmark', errors='ignore')

    # # Plot heatmap
    # plt.figure(figsize=(10, 6))
    # sns.heatmap(df_heat, annot=True, fmt=".2f", cmap='RdYlGn', center=0, cbar_kws={'label': '% Change'})

    # plt.title("Strategy Performance Improvement (% Delta) with Risk Model, rebalance frequency: " + rebalance_freq)
    # plt.ylabel("Strategy")
    # plt.xlabel("Metric")
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # plt.show()

def iterate_time_frame():
    start_year = 2000
    duration_years = 10  # Fixed duration for each range
    symbols = ['SPY', 'AGG']
    rebalance_freq = "weekly"

    all_stats = []
    for start_year in range(start_year, 2016, 5):  # Increment start year by 10
        start_dt = pd.Timestamp(f'{start_year}-01-01 14:30:00', tz=pytz.UTC)
        end_dt = pd.Timestamp(f'{start_year + duration_years - 1}-12-31 23:59:00', tz=pytz.UTC)

        print(f"Running analysis for date range: {start_dt} to {end_dt}")
        run_name = f"TimeFrameTest-{start_dt}-{end_dt}"
        stats = compare_and_plot(start_dt, end_dt, symbols, run_name, rebalance_freq)
        print(stats)
        all_stats.append(stats)
        merged_stats = pd.concat(all_stats, ignore_index=True)
        merged_stats.to_csv("strategy_stats_over_time.csv", index=False)

if __name__ == "__main__":
    start_dt = pd.Timestamp('2015-01-01 14:30:00', tz=pytz.UTC)
    end_dt = pd.Timestamp('2025-02-28 23:59:00', tz=pytz.UTC)
    symbols = ['SPY', 'AGG']
    strategy_runner = StrategyRunner(start_dt, end_dt, symbols, "QLearning1.0", False, False)
    strategy_runner.name ="QLearning1.0-" + strategy_runner._get_run_number()
    
    risk_model = create_risk_model(strategy_runner.get_strategy_factory(), start_dt, end_dt)
    rebalance_freq = "weekly"
    # compare_and_plot(start_dt, end_dt, rebalance_freq)

    strategy_runner.execute_QLearning()
    #iterate_time_frame()

    #print(strategy_runner.execute_sma(38, 39, rebalance_freq, None))
    #print(strategy_runner.execute_custom_cma(38, 39, rebalance_freq, risk_model))

    #print(strategy_runner.execute_rsi(rebalance_freq="daily", risk_model=risk_model))
    #print(strategy_runner.execute_ema(37, 36, "weekly", risk_model))
    #print(strategy_runner.execute_custom_cma(10, 90, "weekly", risk_model))
    #print(strategy_runner.execute_sma(7, 8, "weekly"))
    #print(strategy_runner.execute_sma(7, 8, "weekly", risk_model))
    #print(strategy_runner.execute_wma(7, 8, "weekly"))
    #print(strategy_runner.execute_wma(7, 8, "weekly", risk_model))