# Evaluating the Effectiveness of ES VS DQN in Stock Trading

This project implements and compares two reinforcement learning algorithms for algorithmic stock trading: Evolution Strategy (ES) and Deep Q-Network (DQN). The system downloads historical stock data, trains trading agents, executes trading strategies, and compares performance metrics.

## Project Structure

├── process_stock_data.py  # Download and prepare stock data from Yahoo Finance
├── RLagent_withDQN.py     # Main reinforcement learning implementation
├── gradio_interface.py    # The web interface to react with
├── visualization.py
├── README.md
├── data/                  # Stock price data (CSV files)
│   ├── ABBV.csv
│   ├── CAT.csv
│   ├── COP.csv
│   ├── GOOGL.csv
│   ├── GS.csv
│   └── SBUX.csv
└── pic/
    └── trades/
        ├── CAT_DQN_trades.png
        ├── CAT_ES_trades.png
        ├── SBUX_DQN_trades.png
        ├── SBUX_ES_trades.png
        ├── COP_DQN_trades.png
        ├── COP_ES_trades.png
        ├── ABBV_DQN_trades.png
        ├── ABBV_ES_trades.png
        ├── GS_DQN_trades.png
        ├── GS_ES_trades.png
        ├── GOOGL_DQN_trades.png
        └── GOOGL_ES_trades.png

## How to Use

1. **Download Stock Data**
python process_stock_data.py
This downloads historical data for all six stocks and processes it for use by the trading agents. Or you can just run `python RLagent_withDQN.py` because all six stocks data is stored at the data file.

2. **Train Models and Compare Algorithms**
python RLagent_withDQN.py
This trains both ES and DQN models for each stock, executes trading strategies, and generates performance visualizations and comparisons.

3. **Launch Web Interface**
python gradio_interface.py
Opens an interactive web interface where you can select stocks to analyze, view trading performance, compare algorithms, and visualize transaction history.