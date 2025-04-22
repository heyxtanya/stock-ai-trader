import gradio as gr
import pandas as pd
import os
from PIL import Image
import warnings
from datetime import datetime

from RLagent_withDQN import compare_algorithms

warnings.filterwarnings("ignore")

SAVE_DIR = 'tmp/gradio'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(f'{SAVE_DIR}/pic', exist_ok=True)
os.makedirs(f'{SAVE_DIR}/ticker', exist_ok=True)

transactions_df_es = gr.DataFrame(
    headers=["day", "operate", "price", "investment", "total_balance"],
    label="ES Transaction Records"
)
transactions_df_dqn = gr.DataFrame(
    headers=["day", "operate", "price", "investment", "total_balance"],
    label="DQN Transaction Records"
)

def get_data(ticker, start_date, end_date, progress=gr.Progress()):
    data_folder = f'{SAVE_DIR}/ticker'
    temp_path = f'{data_folder}/{ticker}.csv'
    try:
        progress(0, desc=f"Loading local data for {ticker}...")
        file_path = f"data/{ticker}.csv"
        stock_data = pd.read_csv(file_path, parse_dates=True)
        
        if stock_data.empty:
            raise ValueError("Local data is empty. Please check the file content.")
        
        stock_data.to_csv(temp_path, index=False)
        progress(1.0, desc="Preprocessing completed.")
        return temp_path, f"{ticker} data loaded and formatted successfully!"
    except Exception as e:
        return None, f"Error loading local data: {str(e)}"

def process_and_predict(temp_csv_path, window_size, initial_money, agent_iterations, save_dir):
    if not temp_csv_path:
        return [None] * 15

    try:
        ticker = os.path.splitext(os.path.basename(temp_csv_path))[0]

        es_result, dqn_result = compare_algorithms(
            ticker,
            save_dir,
            window_size=window_size,
            initial_money=initial_money,
            iterations=agent_iterations
        )

        trade_plot_path_es = f"{save_dir}/pic/trades/{ticker}_ES_trades.png"
        trade_plot_path_dqn = f"{save_dir}/pic/trades/{ticker}_DQN_trades.png"

        trades_plot = None
        if os.path.exists(trade_plot_path_es):
            trades_plot = Image.open(trade_plot_path_es)
        elif os.path.exists(trade_plot_path_dqn):
            trades_plot = Image.open(trade_plot_path_dqn)

        transaction_path_es = f"{save_dir}/transactions/{ticker}_ES_transactions.csv"
        transaction_path_dqn = f"{save_dir}/transactions/{ticker}_DQN_transactions.csv"

        transactions_df_es = pd.read_csv(transaction_path_es) if os.path.exists(transaction_path_es) else pd.DataFrame(columns=["day", "operate", "price", "investment", "total_balance"])
        transactions_df_dqn = pd.read_csv(transaction_path_dqn) if os.path.exists(transaction_path_dqn) else pd.DataFrame(columns=["day", "operate", "price", "investment", "total_balance"])

        return [
            [trades_plot],
            es_result['total_profit'],
            es_result['percent_return'],
            es_result['buy_trades'],
            es_result['sell_trades'],
            dqn_result['total_profit'],
            dqn_result['percent_return'],
            dqn_result['buy_trades'],
            dqn_result['sell_trades'],
            transactions_df_es,
            transactions_df_dqn
        ]
    except Exception as e:
        print(f"Error: {str(e)}")
        return [None] * 15

with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("<h2 style='text-align: center;'>📈 RL Trading Dashboard: ES vs. DQN</h2>")
    save_dir_state = gr.State(value=SAVE_DIR)
    temp_csv_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=2):
            ticker_input = gr.Textbox(label="Ticker Symbol (e.g., ABBV)")
        with gr.Column(scale=2):
            start_date = gr.Textbox(label="Start Date", value="2023-01-01")
        with gr.Column(scale=2):
            end_date = gr.Textbox(label="End Date", value="2024-12-30")
        with gr.Column(scale=1):
            fetch_button = gr.Button("Fetch Data")

    with gr.Row():
        status_output = gr.Textbox(label="Status", interactive=False)
    with gr.Row():
        data_file = gr.File(label="Download CSV", visible=True, interactive=False)

    with gr.Tabs():
        with gr.TabItem("Agent Parameters"):
            with gr.Column():
                window_size = gr.Slider(10, 100, value=30, step=5, label="Window Size")
                initial_money = gr.Number(value=10000, label="Initial Capital ($)")
                agent_iterations = gr.Slider(100, 1000, value=500, step=50, label="Agent Iterations")

    train_button = gr.Button("Start Training", interactive=False)

    output_gallery = gr.Gallery(label="Trading Result", columns=1, height="auto")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### ES Metrics")
            gains_output = gr.Number(label="Total Profit ($)")
            return_output = gr.Number(label="Return (%)")
            trades_buy_output = gr.Number(label="Buy Trades")
            trades_sell_output = gr.Number(label="Sell Trades")
        with gr.Column():
            gr.Markdown("### DQN Metrics")
            gains_output_dqn = gr.Number(label="Total Profit ($)")
            return_output_dqn = gr.Number(label="Return (%)")
            trades_buy_output_dqn = gr.Number(label="Buy Trades")
            trades_sell_output_dqn = gr.Number(label="Sell Trades")

    with gr.Row():
        gr.Markdown("### Transaction Records")
    with gr.Row():
        with gr.Tabs():
            with gr.TabItem("ES Transactions"):
                transactions_df_es.render()
            with gr.TabItem("DQN Transactions"):
                transactions_df_dqn.render()

    def update_interface(csv_path):
        return (
            csv_path if csv_path else None,
            gr.update(interactive=bool(csv_path))
        )

    fetch_result = fetch_button.click(
        fn=get_data,
        inputs=[ticker_input, start_date, end_date],
        outputs=[temp_csv_state, status_output]
    )
    fetch_result.then(
        update_interface,
        inputs=[temp_csv_state],
        outputs=[data_file, train_button]
    )

    train_button.click(
        fn=process_and_predict,
        inputs=[
            temp_csv_state,
            window_size,
            initial_money,
            agent_iterations,
            save_dir_state
        ],
        outputs=[
            output_gallery,
            gains_output,
            return_output,
            trades_buy_output,
            trades_sell_output,
            gains_output_dqn,
            return_output_dqn,
            trades_buy_output_dqn,
            trades_sell_output_dqn,
            transactions_df_es,
            transactions_df_dqn
        ]
    )

demo.launch(server_port=7860, share=True)