import os
import pandas as pd
import yfinance as yf
from fredapi import Fred


class SimpleDataLoader:
    def __init__(self, data_dir='./DataLoaderContent'):
        self.data_dir = data_dir
        self.fred_client = Fred('ad0b5643bdb95e2644ae09458c61ec08')
        os.makedirs(data_dir, exist_ok=True)

    def get_yahoo_data(self, ticker, start_date, end_date):
        file_path = os.path.join(self.data_dir, f"{ticker}_{start_date}_{end_date}.csv")
        
        if os.path.exists(file_path):
            print(f"Loading {ticker} data from CSV.")
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        else:
            print(f"Downloading {ticker} data from Yahoo Finance.")
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
            df.index.name = 'Date'

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    "_".join(str(level) for level in col).replace(" ", "_").replace("^", "").strip()
                    for col in df.columns
                ]

            df.to_csv(file_path)

        return df
    
    def get_fred_data(self, series_id, start_date, end_date, df_index):
        file_path = os.path.join(self.data_dir, f"{series_id}_{start_date}_{end_date}.csv")
        
        if os.path.exists(file_path):
            print(f"Loading {series_id} data from CSV.")
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        else:
            data = self.fred_client.get_series(series_id,
                                observation_start=start_date,
                                observation_end=end_date)
            df = pd.DataFrame({df_index:data})
            df.index.name = 'Date'
            df.to_csv(file_path)
        return df
    
    def merge_and_forward_fill(self, left_df, right_df):
        merged = left_df.join(right_df, how='left', on="Date")
        merged.ffill(inplace=True)
        return merged

    def get_data_frame(self, start_date, end_date):
        tickers = ['SPY', 'AAPL', '^VIX']
        data_frames = []

        for ticker in tickers:            
            skip_volume = ['VIX']
            clean_ticker = ticker.replace('^', '')

            adj_col = f'Adj_Close_{clean_ticker}'
            vol_col = f'Volume_{clean_ticker}'  # Clean up symbols
            df = self.get_yahoo_data(ticker, start_date, end_date)

            df = df[[adj_col, vol_col]].copy()
            df.columns = [adj_col, vol_col]

            # Calculate percentage changes
            df[f'{adj_col}_change'] = df[adj_col].pct_change()

            if clean_ticker not in skip_volume:
                df[f'{vol_col}_change'] = df[vol_col].pct_change()

            data_frames.append(df.dropna())
        
        data_frames.append(self.get_fred_data('DGS10', start_date, end_date, '10Y_Treasury').dropna())
        data_frames.append(self.get_fred_data('DGS2',  start_date, end_date, '2Y_Treasury').dropna())
        data_frames.append(self.get_fred_data('BAMLH0A0HYM2', start_date, end_date, 'CPI').dropna())

        merged = data_frames[0]
        for df in data_frames[1:]:
            merged = self.merge_and_forward_fill(merged, df)

        merged = merged.dropna()
        print(merged)
        return merged