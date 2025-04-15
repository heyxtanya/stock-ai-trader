import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings
from visualization import (
    plot_stock_prediction,
    plot_training_loss,
    plot_cumulative_earnings,
    plot_accuracy_comparison
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# Set device for PyTorch (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMModel(nn.Module):
    """
    LSTM Neural Network for stock price prediction
    
    Parameters:
        input_size: Number of input features
        hidden_size: Number of hidden units in LSTM layer
        num_layers: Number of LSTM layers
        output_size: Number of output values (typically 1 for price prediction)
        dropout: Dropout rate for regularization
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer with dropout for regularization
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through the network"""
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get output from the last time step
        out = self.fc(out[:, -1, :])
        return out


def load_stock_data(ticker, data_dir='data'):
    """
    Load stock data from CSV file
    
    Parameters:
        ticker: Stock symbol
        data_dir: Directory containing stock data files
        
    Returns:
        DataFrame with stock data
    """
    file_path = os.path.join(data_dir, f'{ticker}.csv')
    data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    return data


def extract_features_and_target(data):
    """
    Extract features and target variable from stock data
    
    Parameters:
        data: DataFrame with stock data
        
    Returns:
        X: Feature DataFrame
        y: Target Series (percentage change in closing price)
    """
    # Select relevant features for prediction
    features = [
        'Volume', 'Year', 'Month', 'Day', 'MA5', 'MA10', 'MA20', 'RSI', 'MACD',
        'VWAP', 'SMA', 'Std_dev', 'Upper_band', 'Lower_band', 'Relative_Performance', 'ATR',
        'Close_prev', 'Open_prev', 'High_prev', 'Low_prev'
    ]
    
    # Extract features (skip first row as some features need previous data)
    X = data[features].iloc[1:]
    
    # Target is percentage change in closing price
    y = data['Close'].pct_change().iloc[1:]
    
    return X, y


def create_sequences(data, sequence_length):
    """
    Create sequences of data for time series prediction
    
    Parameters:
        data: Input data array
        sequence_length: Length of each sequence
        
    Returns:
        X: Sequences of input data
        y: Target values corresponding to each sequence
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        # Create sequence of length sequence_length
        X.append(data[i:i + sequence_length])
        
        # Target is the value immediately after the sequence
        y.append(data[i + sequence_length])
        
    return np.array(X), np.array(y)


def evaluate_predictions(ticker, data, predictions, test_indices, predicted_values, actual_percentages, save_dir):
    """
    Evaluate and visualize prediction results
    
    Parameters:
        ticker: Stock symbol
        data: Original stock data
        predictions: Dictionary of prediction results
        test_indices: Dates for test data
        predicted_values: Predicted stock prices
        actual_percentages: Actual percentage changes
        save_dir: Directory to save visualization results
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Get actual prices for comparison
    actual_prices = data['Close'].loc[test_indices].values
    predicted_prices = np.array(predicted_values)
    
    # Calculate error metrics
    mse = np.mean((predicted_prices - actual_prices) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predicted_prices - actual_prices))
    accuracy = 1 - np.mean(np.abs(predicted_prices - actual_prices) / actual_prices)
    
    # Compile metrics
    metrics = {'rmse': rmse, 'mae': mae, 'accuracy': accuracy}
    
    # Visualize predictions
    plot_stock_prediction(ticker, test_indices, actual_prices, predicted_prices, metrics, save_dir)
    
    return metrics


def train_and_predict_lstm(ticker, data, X, y, save_dir, n_steps=60, num_epochs=200, batch_size=32, learning_rate=0.001):
    """
    Train LSTM model and make predictions
    
    Parameters:
        ticker: Stock symbol
        data: Original stock data
        X: Feature data
        y: Target data
        save_dir: Directory to save results
        n_steps: Sequence length for LSTM
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        
    Returns:
        predict_result: Dictionary of prediction results
        test_indices: Dates for test data
        predictions: Predicted stock prices
        actual_percentages: Actual percentage changes
    """
    # Data normalization
    scaler_y = MinMaxScaler()
    scaler_X = MinMaxScaler()
    
    # Fit scalers to data
    scaler_y.fit(y.values.reshape(-1, 1))
    y_scaled = scaler_y.transform(y.values.reshape(-1, 1))
    X_scaled = scaler_X.fit_transform(X)

    # Create sequences for time series prediction
    X_sequences, y_sequences = create_sequences(X_scaled, n_steps)
    y_sequences = y_scaled[n_steps-1:-1]  # Align targets with sequences

    # Split data into training and validation sets
    train_ratio = 0.8
    split_index = int(train_ratio * len(X_sequences))
    
    X_val = X_sequences[split_index-n_steps+1:]
    y_val = y_sequences[split_index-n_steps+1:]
    X_train = X_sequences[:split_index]
    y_train = y_sequences[:split_index]

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Create PyTorch datasets and data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, optimizer and learning rate scheduler
    model = LSTMModel(
        input_size=X_train.shape[2],  # Number of features
        hidden_size=50,               # Hidden layer size
        num_layers=2,                 # Number of LSTM layers
        output_size=1                 # Predict one value (price change)
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # Lists to store training and validation losses
    train_losses = []
    val_losses = []

    # Training loop with progress bar
    with tqdm(total=num_epochs, desc=f"Training {ticker}", unit="epoch") as pbar:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0
            
            for inputs, targets in train_loader:
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()

            # Calculate average training loss
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase
            model.eval()
            epoch_val_loss = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    val_loss = criterion(outputs, targets)
                    epoch_val_loss += val_loss.item()

            # Calculate average validation loss
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Update progress bar
            pbar.set_postfix({"Train Loss": avg_train_loss, "Val Loss": avg_val_loss})
            pbar.update(1)
            
            # Update learning rate
            scheduler.step()

    # Visualize training and validation loss
    plot_training_loss(ticker, train_losses, val_losses, save_dir)

    # Make predictions on test data
    model.eval()
    predictions = []
    test_indices = []
    predict_percentages = []
    actual_percentages = []

    # Generate predictions for each time step after training data
    with torch.no_grad():
        for i in range(1 + split_index, len(X_scaled) + 1):
            # Prepare input sequence
            x_input = torch.tensor(
                X_scaled[i - n_steps:i].reshape(1, n_steps, X_train.shape[2]), 
                dtype=torch.float32
            ).to(device)
            
            # Make prediction
            y_pred = model(x_input)
            
            # Convert prediction back to original scale
            y_pred = scaler_y.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
            
            # Calculate predicted price based on percentage change
            predictions.append((1 + y_pred[0][0]) * data['Close'].iloc[i - 2])
            
            # Store date, predicted and actual percentage changes
            test_indices.append(data.index[i - 1])
            predict_percentages.append(y_pred[0][0] * 100)
            actual_percentages.append(y[i - 1] * 100)

    # Visualize cumulative earnings based on predictions vs actual
    plot_cumulative_earnings(ticker, test_indices, actual_percentages, predict_percentages, save_dir)

    # Format prediction results
    predict_result = {str(date): pred / 100 for date, pred in zip(test_indices, predict_percentages)}
    
    return predict_result, test_indices, predictions, actual_percentages


def save_predictions(ticker, test_indices, predictions, save_dir):
    """
    Save prediction results to file
    
    Parameters:
        ticker: Stock symbol
        test_indices: Dates for test data
        predictions: Predicted stock prices
        save_dir: Directory to save results
    """
    # Create DataFrame with predictions
    df = pd.DataFrame({
        'Date': test_indices,
        'Prediction': predictions
    })

    # Create directory if it doesn't exist
    file_path = os.path.join(save_dir, 'predictions', f'{ticker}_predictions.pkl')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save predictions to pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(df, file)

    print(f'Saved predictions for {ticker} to {file_path}')


def predict_stock(ticker_name, stock_data, stock_features, save_dir, epochs=200, batch_size=32, learning_rate=0.001):
    """
    Main function to predict stock prices
    
    Parameters:
        ticker_name: Stock symbol
        stock_data: DataFrame with stock data
        stock_features: Tuple of (X, y) with features and target
        save_dir: Directory to save results
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        
    Returns:
        Dictionary of evaluation metrics
    """
    all_predictions = {}
    prediction_metrics = {}

    print(f"\nProcessing {ticker_name}")
    data = stock_data
    X, y = stock_features
    
    # Train model and make predictions
    predict_result, test_indices, predictions, actual_percentages = train_and_predict_lstm(
        ticker_name, data, X, y, save_dir, 
        num_epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )
    all_predictions[ticker_name] = predict_result
    
    # Evaluate predictions
    metrics = evaluate_predictions(
        ticker_name, data, predict_result, test_indices, 
        predictions, actual_percentages, save_dir
    )
    prediction_metrics[ticker_name] = metrics
    
    # Save predictions for use by trading agent
    save_predictions(ticker_name, test_indices, predictions, save_dir)

    # Save prediction metrics
    os.makedirs(os.path.join(save_dir, 'output'), exist_ok=True)
    metrics_df = pd.DataFrame(prediction_metrics).T
    metrics_df.to_csv(os.path.join(save_dir, 'output', f'{ticker_name}_prediction_metrics.csv'))

    # Visualize accuracy comparison
    plot_accuracy_comparison(prediction_metrics, save_dir)

    # Generate summary report
    summary = {
        'Average Accuracy': np.mean([m['accuracy'] * 100 for m in prediction_metrics.values()]),
        'Average RMSE': metrics_df['rmse'].mean(),
        'Average MAE': metrics_df['mae'].mean()
    }

    # Save summary report
    with open(os.path.join(save_dir, 'output', f'{ticker_name}_prediction_summary.txt'), 'w') as f:
        for key, value in summary.items():
            f.write(f'{key}: {value}\n')

    print("\nPrediction Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    return metrics


if __name__ == "__main__":
    # List of stocks to analyze
    tickers = [
        'GOOGL', # Technology
        'GS',    # Finance
        'ABBV',  # Healthcare
        'COP',   # Energy
        'SBUX',  # Consumer
        'CAT'    # Industrial
    ]

    save_dir = 'results'  # Directory to save results
    
    # Process each stock
    for ticker_name in tickers:
        # Load and prepare data
        stock_data = load_stock_data(ticker_name)
        stock_features = extract_features_and_target(stock_data)
        
        # Train model and make predictions
        predict_stock(
            ticker_name=ticker_name,
            stock_data=stock_data,
            stock_features=stock_features,
            save_dir=save_dir
        )