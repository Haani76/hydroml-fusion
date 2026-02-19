"""
LSTM Deep Learning Model for Streamflow Forecasting
Sequence-to-sequence architecture
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class StreamflowDataset(Dataset):
    """
    Dataset for LSTM training
    Creates sequences of input features and target streamflow
    """
    
    def __init__(self, features, target, sequence_length=30):
        """
        Args:
            features: Input features (N, n_features)
            target: Target streamflow (N,)
            sequence_length: Length of input sequences
        """
        self.features = torch.FloatTensor(features)
        self.target = torch.FloatTensor(target)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence of features
        x = self.features[idx:idx + self.sequence_length]
        # Get corresponding target (one step ahead)
        y = self.target[idx + self.sequence_length]
        return x, y


class LSTMModel(nn.Module):
    """
    LSTM Neural Network for Streamflow Prediction
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Args:
            input_size: Number of input features
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input sequences (batch_size, seq_length, input_size)
            
        Returns:
            Predictions (batch_size, 1)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take output from last time step
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class LSTMTrainer:
    """
    Training and evaluation handler for LSTM model
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            predictions = self.model(X_batch)
            loss = self.criterion(predictions.squeeze(), y_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        """Evaluate on validation set"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = self.criterion(predictions.squeeze(), y_batch)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=50, verbose=True):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            verbose: Print progress
        """
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.train_losses, self.val_losses
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input features (numpy array or tensor)
            
        Returns:
            Predictions (numpy array)
        """
        self.model.eval()
        
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        
        X = X.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X)
        
        return predictions.cpu().numpy().flatten()