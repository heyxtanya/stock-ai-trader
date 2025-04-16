import numpy as np
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
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
        model: Neural network model (for ES) or None (for DQN)
        window_size: Number of price points to consider
        price_history: Historical price data
        skip: Steps to skip between actions
        initial_capital: Starting money
        ticker: Stock symbol
        save_dir: Directory to save results
        algorithm: Trading algorithm to use ('ES' or 'DQN')
    """
    # ES Hyperparameters
    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03
    
    # DQN Hyperparameters
    GAMMA = 0.95       
    EPSILON = 0.1      
    BATCH_SIZE = 32    
    MEMORY_SIZE = 1000

    def __init__(self, model, window_size, price_history, skip, initial_capital, ticker, save_dir, algorithm='ES'):
        self.window_size = window_size
        self.price_history = price_history
        self.skip = skip
        self.initial_capital = initial_capital
        self.ticker = ticker
        self.save_dir = save_dir
        self.algorithm = algorithm
        
        if self.algorithm == 'ES':
            self.model = model
            self.evolution_strategy = DeepEvolutionStrategy(
                self.model.get_weights(),
                self.calculate_reward,
                self.POPULATION_SIZE,
                self.SIGMA,
                self.LEARNING_RATE,
            )
        elif self.algorithm == 'DQN':
            self.state_size = window_size
            self.action_size = 3  # hold, buy, sell
            self.memory = deque(maxlen=self.MEMORY_SIZE)
            self.epsilon = self.EPSILON
            
            self.dqn_model = self._build_dqn_model()
            self.dqn_target_model = self._build_dqn_model()
            self.dqn_target_model.load_state_dict(self.dqn_model.state_dict())
            self.optimizer = optim.Adam(self.dqn_model.parameters(), lr=0.001)

    def _build_dqn_model(self):
        """Eastablish the DQN Modle"""
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_size)
        )
        return model

    def select_action(self, state):
        """Select action based on current state"""
        if self.algorithm == 'ES':
            # ES decision logic
            decision = self.model.predict(np.array(state))
            return np.argmax(decision[0])
        elif self.algorithm == 'DQN':
            # DQN decision logic
            if np.random.rand() <= self.epsilon:
                return np.random.randint(self.action_size) # Exploration: random action
            else:
                # Exploitation: select best action based on Q values
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.dqn_model(state_tensor)
                return torch.argmax(q_values).item()

    def get_state(self, time_index):
        """Extract state representation"""
        window_size = self.window_size + 1
        start_index = time_index - window_size + 1
        
        # Handle boundary conditions
        if start_index >= 0:
            price_window = self.price_history[start_index: time_index + 1]
        else:
            # Fill with initial price
            padding = -start_index * [self.price_history[0]]
            price_window = padding + self.price_history[0: time_index + 1]
            
        # Calculate price changes
        price_changes = []
        for i in range(window_size - 1):
            price_changes.append(price_window[i + 1] - price_window[i])
        
        # Return state in format based on algorithm
        if self.algorithm == 'ES':
            return np.array([price_changes])
        else:  # DQN
            return np.array(price_changes)

    def calculate_reward(self, weights):
        """Evaluate trading performance"""
        # Only used for ES algorithm
        capital = self.initial_capital
        starting_capital = capital
        self.model.weights = weights
        state = self.get_state(0)
        inventory = []  
        
        # Simulate trading
        for t in range(0, len(self.price_history) - 1, self.skip):
            action = self.select_action(state)
            next_state = self.get_state(t + 1)
            current_price = self.price_history[t]

            # Buy
            if action == 1 and capital >= current_price:
                inventory.append(current_price)
                capital -= current_price

            # Sell
            elif action == 2 and len(inventory) > 0:
                bought_price = inventory.pop(0)
                capital += current_price

            state = next_state
            
        # Calculate percentage return
        return ((capital - starting_capital) / starting_capital) * 100

    def remember(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Learn from experience replay buffer"""
        if len(self.memory) < self.BATCH_SIZE:
            return 0
        
        # Random sampling
        minibatch = random.sample(self.memory, self.BATCH_SIZE)
        
        # Extract sample data
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for state, action, reward, next_state, done in minibatch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        # Convert to PyTorch tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        
        # Calculate current Q values
        q_values = self.dqn_model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculate target Q values
        with torch.no_grad():
            next_q_values = self.dqn_target_model(next_states).max(1)[0]
            targets = rewards + (1 - dones) * self.GAMMA * next_q_values
        
        # Calculate loss and optimize
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_model(self):
        """Update target network"""
        self.dqn_target_model.load_state_dict(self.dqn_model.state_dict())

    def train(self, iterations=200, checkpoint=10):
        """Train the agent"""
        if self.algorithm == 'ES':
            # ES Training
            self.evolution_strategy.train(iterations, checkpoint)
        elif self.algorithm == 'DQN':
            # DQN Training 
            print(f"Train DQN Agent: {self.ticker}...")
            start_time = time.time()
            
            for episode in range(iterations):
                # Reset environment state
                state = self.get_state(0)
                total_reward = 0
                inventory = []
                capital = self.initial_capital
                
                # Trading loop
                for t in range(0, len(self.price_history) - 1, self.skip):
                    # Select action
                    action = self.select_action(state)
                    next_state = self.get_state(t + 1)
                    current_price = self.price_history[t]
                    done = (t == len(self.price_history) - 2)
                    
                    # Execute action and calculate reward
                    reward = 0
                    
                    # buy
                    if action == 1 and capital >= current_price:
                        inventory.append(current_price)
                        capital -= current_price
                        reward = 0  # No immediate reward when buying
                    
                    # Sell
                    elif action == 2 and len(inventory) > 0:
                        bought_price = inventory.pop(0)
                        capital += current_price
                        reward = current_price - bought_price   # Reward is profit
                    
                    # hold
                    else:
                        reward = -0.1  # Small negative reward to encourage action
                    
                    # If final step, calculate additional reward based on total assets
                    if done:
                        # Calculate total assets (cash + inventory)
                        total_assets = capital
                        for stock_price in inventory:
                            total_assets += self.price_history[-1]
                        
                        # Calculate reward based on total return
                        final_return = ((total_assets - self.initial_capital) / self.initial_capital) * 100
                        reward += final_return
                    
                    # Store experience and learn
                    self.remember(state, action, reward, next_state, done)
                    self.replay()
                    
                    # Accumulate reward
                    total_reward += reward
                    
                    # Transition to next state
                    state = next_state
                    
                    if done:
                        break
                
                # Periodically update target network
                if episode % 10 == 0:
                    self.update_target_model()
                
                # Print training progress
                if (episode + 1) % checkpoint == 0:
                    print(f"Episode: {episode + 1}/{iterations}, Reward: {total_reward:.2f}")
            
            print(f'Training completed in {time.time() - start_time:.2f} seconds')



    def execute_strategy(self, save_dir):
        """Execute trading strategy and record results"""
        # For DQN, temporarily set epsilon to 0 (no exploration)
        if self.algorithm == 'DQN':
            saved_epsilon = self.epsilon
            self.epsilon = 0
        
        # Initialize
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

            # Buy
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

            # Sell
            elif action == 2 and len(inventory) > 0:
                bought_price = inventory.pop(0)
                capital += current_price
                sell_timestamps.append(t)
                
                # Calculate trade return
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

        # Restore DQN's epsilon value
        if self.algorithm == 'DQN':
            self.epsilon = saved_epsilon

        df_transaction = pd.DataFrame(transaction_history)
        os.makedirs(f'{save_dir}/transactions', exist_ok=True)
        df_transaction.to_csv(f'{save_dir}/transactions/{self.ticker}_{self.algorithm}_transactions.csv', index=False)

        # Calculate overall performance
        total_return = ((capital - starting_capital) / starting_capital) * 100
        total_profit = capital - starting_capital
        
        return buy_timestamps, sell_timestamps, total_profit, total_return


def process_stock(ticker, save_dir, window_size=30, initial_money=10000, iterations=200, algorithm='ES'):
    """Process a single stock using specified algorithm"""
    try:
        # Load predicted price data
        df = pd.read_pickle(f'{save_dir}/predictions/{ticker}_predictions.pkl')
        print(f"\nUse {algorithm} to handle {ticker}")
        price_data = df.Prediction.values.tolist()

        # Configure trading parameters
        skip = 1  # Process every day

        if algorithm == 'ES':
            # Create ES model and agent
            model = TradingModel(input_size=window_size, hidden_size=200, output_size=3)
            agent = TradingAgent(
                model=model, 
                window_size=window_size, 
                price_history=price_data, 
                skip=skip, 
                initial_capital=initial_money, 
                ticker=ticker, 
                save_dir=save_dir,
                algorithm='ES'
            )
        elif algorithm == 'DQN':
            # Create DQN agent (no separate model needed)
            agent = TradingAgent(
                model=None,  # DQN doesn't need external model
                window_size=window_size, 
                price_history=price_data, 
                skip=skip, 
                initial_capital=initial_money, 
                ticker=ticker, 
                save_dir=save_dir,
                algorithm='DQN'
            )
        
    
        print(f"Training {algorithm} agent...")
        agent.train(iterations=iterations, checkpoint=10)


        print(f"Execute {algorithm}Trading Strategy...")
        buy_times, sell_times, total_profit, percent_return = agent.execute_strategy(save_dir)

        # Visualize results
        plot_trading_result(f"{ticker}_{algorithm}", price_data, buy_times, sell_times, total_profit, percent_return, save_dir)
        
        return {
            'total_profit': total_profit,
            'percent_return': percent_return,
            'buy_trades': len(buy_times),
            'sell_trades': len(sell_times),
            'algorithm': algorithm
        }
        
    except Exception as e:
        print(f"File to handle{ticker}: {e}")
        return None


def compare_algorithms(ticker, save_dir, window_size=30, initial_money=10000, iterations=200):
    """Compare ES and DQN performance on the same stock"""
    # Run Evolution Strategy
    print("\n Run the Evolution Strategy...")
    es_result = process_stock(ticker, save_dir, window_size, initial_money, iterations, algorithm='ES')
    
    # Run DQN
    print("\nRun the Deep Q-Learning...")
    dqn_result = process_stock(ticker, save_dir, window_size, initial_money, iterations, algorithm='DQN')
    
    # Compare
    if es_result and dqn_result:
        print(f"\n=== {ticker} Compare two Strategy/ Algorithm ===")
        print(f"Evolution Strategy: Gain={es_result['percent_return']:.2f}%, Profit=${es_result['total_profit']:.2f}")
        print(f"Deep Q-Learning:    Gain={dqn_result['percent_return']:.2f}%, Profit=${dqn_result['total_profit']:.2f}")
        
        plot_algorithm_comparison(ticker, es_result, dqn_result, save_dir)
    
    return es_result, dqn_result


def plot_algorithm_comparison(ticker, es_result, dqn_result, save_dir):
    """Plot algorithm comparison chart"""
    metrics = ['percent_return', 'total_profit', 'buy_trades', 'sell_trades']
    labels = ['Return (%)', 'Profit ($)', 'Buy Trades', 'Sell Trades']
    
    plt.figure(figsize=(12, 8))
    
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        plt.subplot(2, 2, i+1)
        data = [es_result[metric], dqn_result[metric]]
        plt.bar(['ES', 'DQN'], data)
        plt.title(f'{ticker} - {label}')
        plt.ylabel(label)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.join(save_dir, 'comparisons'), exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'comparisons', f'{ticker}_algorithm_comparison.png'))
    plt.close()


def main():
    """Main function - Run trading system and compare different algorithms"""
    tickers = [
        'GOOGL', # Technology
        'GS',    # Finance
        'ABBV',  # Healthcare
        'COP',   # Energy
        'SBUX',  # Consumer
        'CAT'    # Industrial
    ]
    save_dir = 'results'
    
    # store results
    all_results = {}
    for ticker in tickers:
        es_result, dqn_result = compare_algorithms(ticker, save_dir)
        all_results[ticker] = {
            'ES': es_result,
            'DQN': dqn_result
        }

    print("\n=== Two Strategies' Comparsion Summary ===")
    print("| Stock   | ES Gain  | DQN Gain | Difference |")
    print("|---------|----------|----------|------------|")
    
    for ticker in tickers:
        if ticker in all_results and all_results[ticker]['ES'] and all_results[ticker]['DQN']:
            es_return = all_results[ticker]['ES']['percent_return']
            dqn_return = all_results[ticker]['DQN']['percent_return']
            diff = dqn_return - es_return
            better = "DQN" if diff > 0 else "ES"
            print(f"| {ticker:6} | {es_return:8.2f}% | {dqn_return:8.2f}% | {abs(diff):4.2f}% ({better}) |")


if __name__ == "__main__":
    main()