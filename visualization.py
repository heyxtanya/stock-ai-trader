import matplotlib.pyplot as plt
import os

def plot_trading_result(ticker, close_prices, states_buy, states_sell, total_gains, invest, save_dir):
    plt.figure(figsize=(15, 5))
    plt.plot(close_prices, color='r', lw=2.)
    plt.plot(close_prices, '^', markersize=10, color='m', label='buying signal', markevery=states_buy)
    plt.plot(close_prices, 'v', markersize=10, color='k', label='selling signal', markevery=states_sell)
    plt.title(f'{ticker} total gains ${total_gains:.2f}, total investment {invest:.2f}%')
    plt.legend()
    
    trades_dir = os.path.join(save_dir, 'pic/trades')
    os.makedirs(trades_dir, exist_ok=True)
    save_path = os.path.join(trades_dir, f'{ticker}_trades.png')
    plt.savefig(save_path)
    plt.close()
    
    return save_path