import os
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def prepare_data(data):
    '''
    Prepare stock data for RL agent.
        data: Stock dataframe with 'Open', 'High', 'Low', 'Close', 'Volume' columns.
        return: Data with date information.
    '''
    # Setting index to datetime
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    
    return data

def main():
    '''
    Main function to download stock data and save to CSV for RL agent.
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
    START = '2023-01-01'
    END = '2024-12-31'
    
    # Set up data folder
    data_folder = 'data'
    os.makedirs(data_folder, exist_ok=True)
    
    # Download and process data, save to CSV
    print("--------Start Downloading Data--------")
    for ticker in tickers:
        try:
            # Download data from Yahoo Finance
            data = yf.download(ticker, start=START, end=END)
            
            # Add date information
            data_prepared = prepare_data(data)
            
            # Save to CSV
            data_prepared.to_csv(f'{data_folder}/{ticker}.csv')
            print(f"Data of {ticker} successfully downloaded and processed.")
        except Exception as e:
            print(f"{ticker} has error in process: {str(e)}")
    print("--------All Data Downloaded and Processed--------")

if __name__ == "__main__":
    main()