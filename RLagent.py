import numpy as np
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import random
from visualization import plot_trading_result
sns.set()

class DeepEvolutionStrategy:
    """
    Deep Evolution Strategy for optimizing trading model weights
    
    Parameters:
        weights: Initial model weights
        reward_function: Function to evaluate performance
        population_size: Number of individuals in population
        sigma: Standard deviation for weight perturbation
        learning_rate: Rate of weight updates
    """
    def __init__(self, weights, reward_function, population_size, sigma, learning_rate):
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def get_weights(self):
        """Return current model weights"""
        return self.weights        

    def generate_perturbed_weights(self, weights, perturbation):
        """Generate weights with random perturbations"""
        perturbed_weights = []
        for i, p in enumerate(perturbation):
            perturbed_weights.append(weights[i] + self.sigma * p)
        return perturbed_weights

    def train(self, epochs=100, print_frequency=1):
        """
        Train the model using evolutionary strategy
        
        Parameters:
            epochs: Number of training iterations
            print_frequency: How often to print progress
        """
        start_time = time.time()
        for epoch in range(epochs):
            # Generate population of random perturbations
            population = []
            rewards = np.zeros(self.population_size)
            
            for k in range(self.population_size):
                perturbation = []
                for weight in self.weights:
                    perturbation.append(np.random.randn(*weight.shape))
                population.append(perturbation)
            
            # Evaluate each individual in population
            for k in range(self.population_size):
                perturbed_weights = self.generate_perturbed_weights(self.weights, population[k])
                rewards[k] = self.reward_function(perturbed_weights)
            
            # Normalize rewards for stable training
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
            
            # Update weights based on performance
            for index, weight in enumerate(self.weights):
                perturbations = np.array([p[index] for p in population])
                self.weights[index] = (
                    weight
                    + self.learning_rate
                    / (self.population_size * self.sigma)
                    * np.dot(perturbations.T, rewards).T
                )
                
            # Print progress
            if (epoch + 1) % print_frequency == 0:
                print(f'Epoch {epoch + 1}. Reward: {self.reward_function(self.weights):.4f}')
                
        print(f'Training completed in {time.time() - start_time:.2f} seconds')


class TradingModel:
    """
    Simple neural network model for trading decisions
    
    Parameters:
        input_size: Size of input features (window size)
        hidden_size: Size of hidden layer
        output_size: Number of possible actions (hold, buy, sell)
    """
    def __init__(self, input_size, hidden_size, output_size):
        self.weights = [
            np.random.randn(input_size, hidden_size),    
            np.random.randn(hidden_size, output_size),   
            np.random.randn(1, hidden_size),             
        ]

    def predict(self, inputs):
        """Forward pass to predict action probabilities"""
        hidden = np.dot(inputs, self.weights[0]) + self.weights[2] 
        output = np.dot(hidden, self.weights[1])
        return output

    def get_weights(self):
        """Return model weights"""
        return self.weights

    def set_weights(self, weights):
        """Set model weights"""
        self.weights = weights


class TradingAgent:
    """
    Trading agent that learns to buy/sell stocks
    
    Parameters:
        model: Neural network model
        window_size: Number of price points to consider
        price_history: Historical price data
        skip: Steps to skip between actions
        initial_capital: Starting money
        ticker: Stock symbol
        save_dir: Directory to save results
    """
    # Hyperparameters
    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03

    def __init__(self, model, window_size, price_history, skip, initial_capital, ticker, save_dir):
        self.model = model
        self.window_size = window_size
        self.price_history = price_history
        self.skip = skip
        self.initial_capital = initial_capital
        self.ticker = ticker
        self.save_dir = save_dir
        
        # Initialize evolution strategy
        self.evolution_strategy = DeepEvolutionStrategy(
            self.model.get_weights(),
            self.calculate_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )

    def select_action(self, state):
        """Choose action (0=hold, 1=buy, 2=sell) based on current state"""
        decision = self.model.predict(np.array(state))
        return np.argmax(decision[0])

    def get_state(self, time_index):
        """Extract price changes for the window ending at time_index"""
        window_size = self.window_size + 1
        start_index = time_index - window_size + 1
        
        # Handle cases where we need data before the beginning
        if start_index >= 0:
            price_window = self.price_history[start_index: time_index + 1]
        else:
            # Pad with initial price
            padding = -start_index * [self.price_history[0]]
            price_window = padding + self.price_history[0: time_index + 1]
            
        # Calculate price changes
        price_changes = []
        for i in range(window_size - 1):
            price_changes.append(price_window[i + 1] - price_window[i])
            
        return np.array([price_changes])

    def calculate_reward(self, weights):
        """Evaluate trading performance with given weights"""
        # Setup simulation
        capital = self.initial_capital
        starting_capital = capital
        self.model.weights = weights
        state = self.get_state(0)
        inventory = []  # Stocks owned
        
        # Simulate trading
        for t in range(0, len(self.price_history) - 1, self.skip):
            action = self.select_action(state)
            next_state = self.get_state(t + 1)
            current_price = self.price_history[t]

            # Buy action
            if action == 1 and capital >= current_price:
                inventory.append(current_price)
                capital -= current_price

            # Sell action
            elif action == 2 and len(inventory) > 0:
                bought_price = inventory.pop(0)
                capital += current_price

            state = next_state
            
        # Calculate percentage return
        return ((capital - starting_capital) / starting_capital) * 100

    def train(self, iterations, checkpoint):
        """Train the trading agent"""
        self.evolution_strategy.train(iterations, checkpoint)

    def execute_strategy(self, save_dir):
        """Run the trained strategy and record transactions"""
        # Setup
        capital = self.initial_capital
        starting_capital = capital
        state = self.get_state(0)
        buy_timestamps = []
        sell_timestamps = []
        inventory = []
        transaction_history = []

        # Execute trading strategy
        for t in range(0, len(self.price_history) - 1, self.skip):
            action = self.select_action(state)
            next_state = self.get_state(t + 1)
            current_price = self.price_history[t]

            # Buy action
            if action == 1 and capital >= current_price:
                inventory.append(current_price)
                capital -= current_price
                buy_timestamps.append(t)
                transaction_history.append({
                    'day': t,
                    'operate': 'buy',
                    'price': current_price,
                    'investment': 0,
                    'total_balance': capital
                })

            # Sell action
            elif action == 2 and len(inventory) > 0:
                bought_price = inventory.pop(0)
                capital += current_price
                sell_timestamps.append(t)
                
                # Calculate return on this trade
                try:
                    trade_return = ((current_price - bought_price) / bought_price) * 100
                except:
                    trade_return = 0
                    
                transaction_history.append({
                    'day': t,
                    'operate': 'sell',
                    'price': current_price,
                    'investment': trade_return,
                    'total_balance': capital
                })

            state = next_state

        # Save transaction history
        df_transaction = pd.DataFrame(transaction_history)
        os.makedirs(f'{save_dir}/transactions', exist_ok=True)
        df_transaction.to_csv(f'{save_dir}/transactions/{self.ticker}_transactions.csv', index=False)

        # Calculate overall performance
        total_return = ((capital - starting_capital) / starting_capital) * 100
        total_profit = capital - starting_capital
        
        return buy_timestamps, sell_timestamps, total_profit, total_return


def process_stock(ticker, save_dir, window_size=30, initial_money=10000, iterations=200):
    """Process a single stock with the trading agent"""
    try:
        # Load predicted price data
        df = pd.read_pickle(f'{save_dir}/predictions/{ticker}_predictions.pkl')
        print(f"\nProcessing {ticker}")
        price_data = df.Prediction.values.tolist()

        # Configure trading parameters
        skip = 1  # Process every day

        # Create model and agent
        model = TradingModel(input_size=window_size, hidden_size=200, output_size=3)
        agent = TradingAgent(
            model=model, 
            window_size=window_size, 
            price_history=price_data, 
            skip=skip, 
            initial_capital=initial_money, 
            ticker=ticker, 
            save_dir=save_dir
        )
        
        # Train the agent
        print(f"Training agent for {ticker}...")
        agent.train(iterations=iterations, checkpoint=10)

        # Execute trading strategy
        print(f"Executing trading strategy for {ticker}...")
        buy_times, sell_times, total_profit, percent_return = agent.execute_strategy(save_dir)

        # Visualize results
        plot_trading_result(ticker, price_data, buy_times, sell_times, total_profit, percent_return, save_dir)
        
        # Return performance metrics
        return {
            'total_profit': total_profit,
            'percent_return': percent_return,
            'buy_trades': len(buy_times),
            'sell_trades': len(sell_times)
        }
        
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None


def main():
    """Main function to run the trading system on multiple stocks"""
    # Stock portfolio to analyze
    tickers = [
        'GOOGL', # Technology
        'GS',    # Finance
        'ABBV',  # Healthcare
        'COP',   # Energy
        'SBUX',  # Consumer
        'CAT'    # Industrial
    ]
    save_dir = 'results'
    
    # Process each stock
    results = {}
    for ticker in tickers:
        result = process_stock(ticker, save_dir)
        if result:
            results[ticker] = result
    
    # Print summary
    print("\n=== Trading Results Summary ===")
    for ticker, metrics in results.items():
        print(f"{ticker}: Profit=${metrics['total_profit']:.2f} ({metrics['percent_return']:.2f}%)")


if __name__ == "__main__":
    main()