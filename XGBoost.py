'''
Author: Zhenlei Song
Email: songzl@tamu.edu
Purpose: This project is for the I-GUIDE Summer School 2024.

This source code file trains an XGBoost model to predict the impact of housing units.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb

from config import BASE_DIR, DATASET_DIR

if __name__ == "__main__":
    # load data
    file_path = f"{DATASET_DIR}/dataset.csv"
    df = pd.read_csv(file_path)

    # transform all columns to numeric and handle values that can't be converted
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # fill NaN values
    df.fillna(df.mean(), inplace=True)

    # separate features and target variable
    features = df.drop(columns=['Housing-Unit-Impact'])  
    targets = df['Housing-Unit-Impact']

    # standardize the features and target variable
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    features = feature_scaler.fit_transform(features)
    targets = target_scaler.fit_transform(targets.values.reshape(-1, 1)).flatten()

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    # create XGBoost regression model
    xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10000, learning_rate=0.05, reg_lambda=1)

    # early stopping setup
    early_stopping_rounds = 10
    eval_set = [(X_train, y_train), (X_test, y_test)]

    # train the XGBoost model and record the loss every 5 epochs
    xgboost_model.fit(X_train, y_train, eval_metric="rmse", eval_set=eval_set, early_stopping_rounds=early_stopping_rounds, verbose=5)

    # XGBoost prediction
    y_pred_xgb = xgboost_model.predict(X_test)

    # calculate XGBoost mean squared error
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    print(f'XGBoost Test MSE: {mse_xgb:.4f}')

    # save the XGBoost model
    xgboost_model.save_model(f"{BASE_DIR}/models/xgboost_model.json")
