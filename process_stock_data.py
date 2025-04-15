import os
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def tech_indicators(data, start=None, end=None):
    '''
    Calculate technical indicators for stock data.
        data: Stock dataframe with 'Open', 'High', 'Low', 'Close', 'Volume' columns.
        start: Start date for relative performance calculation.
        end: End date for relative performance calculation.
        return: Data with technical indicators.
    '''
    # Setting index to datetime
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    
    # Moving Averages in different periods
    data['MA5'] = data['Close'].shift(1).rolling(window=5).mean()
    data['MA10'] = data['Close'].shift(1).rolling(window=10).mean()
    data['MA20'] = data['Close'].shift(1).rolling(window=20).mean()
    
    # RSI
    cal_delta = data['Close'].diff()
    gain = cal_delta.clip(lower=0)
    loss = -cal_delta.clip(upper=0)
    cal_rs = gain.rolling(window=14).mean() / loss.rolling(window=14).mean()
    data['RSI'] = 100 - (100 / (1 + cal_rs))
    
    # MACD
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    
    # VWAP
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    # Bollinger Bands
    data['SMA'] = data['Close'].rolling(window=20).mean()
    data['Std_dev'] = data['Close'].rolling(window=20).std()
    data['Upper_band'] = data['SMA'] + 2 * data['Std_dev']
    data['Lower_band'] = data['SMA'] - 2 * data['Std_dev']
    
    # Compare with SPY(S&P 500 ETF) as relative performance
    if start and end:
        compare = yf.download('SPY', start=start, end=end)['Close']
        data['Relative_Performance'] = (data['Close'] / compare.values) * 100
    
    # ROC
    data['ROC'] = data['Close'].pct_change(periods=1) * 100
    
    # ATR
    range = pd.concat([data['High'] - data['Low'], abs(data['High'] - data['Close'].shift(1)), abs(data['Low'] - data['Close'].shift(1))], axis=1).max(axis=1)
    data['ATR'] = range.rolling(window=14).mean()
    
    # Previous day prices
    data[['Close_prev', 'Open_prev', 'High_prev', 'Low_prev']] = data[['Close', 'Open', 'High', 'Low']].shift(1)
    
    # Dropping NaN values
    data = data.dropna()
    
    return data

def format_data(file_name):
    '''Function to format the downloaded CSV file.
        file_name: Name of the CSV file to format.
    '''
    df = pd.read_csv(file_name) 
    df = df.drop([0, 1]).reset_index(drop=True) 
    df = df.rename(columns={'Price': 'Date'})    
    df.to_csv(file_name, index=False)

def main():
    '''
    Main function to download stock data, calculate technical indicators, and save to CSV.
    '''
    # Stocks to download
    tickers = [
        'GOOGL', # Technology
        'GS',    # Finance
        'ABBV',  # Healthcare
        'COP',   # Energy
        'SBUX',  # Consumer
        'CAT'    # Industrial
    ]

    # Start and end dates for data download
    START = '2024-01-01'
    END = '2024-12-31'
    
    # Set up data folder
    data_folder = 'data'
    os.makedirs(data_folder, exist_ok=True)
    
    # Download and process data, save to CSV
    print("--------Start Downloading Data--------")
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=START, end=END)
            data_with_indicator = tech_indicators(data, START, END)
            data_with_indicator.to_csv(f'{data_folder}/{ticker}.csv')
            format_data(f'{data_folder}/{ticker}.csv')
            print(f"Data of {ticker} sussessfully downloaded and processed.")
        except Exception as e:
            print(f"{ticker} has error in process: {str(e)}")
    print("--------All Data Downloaded and Processed--------")

if __name__ == "__main__":
    main()