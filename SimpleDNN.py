'''
Author: Zhenlei Song
Email: songzl@tamu.edu
Purpose: This project is for the I-GUIDE Summer School 2024.

This source code file trains a simple feedforward neural network model to predict the impact of housing units.
'''
import geopandas as gpd
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error

from config import BASE_DIR, DATASET_DIR

# Define a simple feedforward neural network
class SimpleDNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleDNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output(x)
        return x

if __name__ == "__main__":
    # Load the data
    gdf = gpd.read_file(f"{DATASET_DIR}/results.geojson")
    df = gdf.drop(columns=['geometry', 'GeoId'])

    # Transform all columns to numeric and handle values that can't be converted
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NaN values
    df.fillna(df.mean(), inplace=True)

    # Separate features and target variable
    features = df.drop(columns=['Housing-Unit-Impact'])
    targets = df['Housing-Unit-Impact']

    # Standardize the features and target variable
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    features = feature_scaler.fit_transform(features)
    targets = target_scaler.fit_transform(targets.values.reshape(-1, 1)).flatten()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    input_dim = X_train.shape[1]
    # Create a simple feedforward neural network model
    model = SimpleDNN(input_dim)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # 添加L2正则化

    # Convert the data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Create a DataLoader for the training data
    batch_size = 128
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Early stopping setup
    early_stopping_patience = 20
    best_loss = float('inf')
    patience_counter = 0

    # Train the model
    num_epochs = 500
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            # Calculate the training loss
            train_loss = total_loss / len(train_loader)
            
            # Calculate the test loss
            model.eval()
            with torch.no_grad():
                predictions = model(X_test_tensor)
                test_loss = criterion(predictions, y_test_tensor).item()
            model.train()
            
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

            # Check for early stopping
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    # Calculate the test MSE
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        mse = mean_squared_error(y_test_tensor, predictions)
        print(f'Test MSE: {mse:.4f}')

    # Save the model
    torch.save(model.state_dict(), f"models/dnn_model.pth")
