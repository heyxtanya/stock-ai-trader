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

    # Ensure datetime index
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')

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
        try:
            # Download SPY data
            spy_data = yf.download('SPY', start=start, end=end)
            
            # Explicitly extract Close column
            if isinstance(spy_data, pd.DataFrame) and 'Close' in spy_data.columns:
                compare = spy_data['Close'].copy()  
                # Ensure it's a Series not a DataFrame
                if isinstance(compare, pd.DataFrame):
                    compare = compare.iloc[:, 0]
                    
                # Reindex to match the data index
                compare = compare.reindex(data.index, method='ffill')
                
                # Calculate relative performance
                data['Relative_Performance'] = (data['Close'] / compare) * 100
                print("Succssfully Cacluated Relative_Performance")
            else:
                raise ValueError("Cannot get the Close clo from SPY")
        except Exception as e:
            print(f"SPY Failed to Download: {str(e)}")
            data['Relative_Performance'] = 100  
        
    
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

def save_clean_data(df, file_path):
    """
    Save data with proper format:
    1. Keep Date as index
    2. Ensure numeric types
    3. Standardize CSV format
    """
    # Force Date column as index if not already datetime indexed
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('Date')
    
    # Clean invalid characters from numeric columns and convert to float
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            # Remove currency symbols and commas
            df[col] = df[col].replace(r'[\$,]', '', regex=True).astype(float)

    
    # Save to CSV with Date index and proper formatting
    df.to_csv(file_path, index=True, index_label='Date')
    print(f"[√] right data without extra headers → {os.path.abspath(file_path)}")

# 不要了，因为 后续跑lstm 要求读 Date 列作为索引，但此函数生成的格式不匹配
# def format_data(file_name):
#     '''Function to format the downloaded CSV file.
#         file_name: Name of the CSV file to format.
#     '''
#     df = pd.read_csv(file_name) 
#     df = df.drop([0, 1]).reset_index(drop=True) 
#     df = df.rename(columns={'Price': 'Date'})    
#     df.to_csv(file_name, index=False)

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
            # Save to individual files
            file_path = os.path.join(data_folder, f'{ticker}.csv') 
            save_clean_data(data_with_indicator, file_path)
            
            # Verify file creation
            if os.path.exists(file_path):
                print(f"[√] {ticker} Raw Data has saved to file → {os.path.abspath(file_path)}")
            else:
                print(f"[×] {ticker} Raw Data failed to save")

        except Exception as e:
            print(f"{ticker} has error in process: {str(e)}")
    print("--------All Data Downloaded and Processed--------")

if __name__ == "__main__":
    main()