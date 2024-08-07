'''
Author: Zhenlei Song
Email: songzl@tamu.edu
Purpose: This project is for the I-GUIDE Summer School 2024.

This source code file trains a CatBoost regression model to predict the impact of housing units.
'''

import pandas as pd

from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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
    features = df.drop(columns=['Housing-Unit-Impact'])  # 将'target_column'替换为实际的目标列名
    targets = df['Housing-Unit-Impact']  # 将'target_column'替换为实际的目标列名

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    features = feature_scaler.fit_transform(features)
    targets = target_scaler.fit_transform(targets.values.reshape(-1, 1)).flatten()

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    # create CatBoost regression model
    catboost_model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=10, loss_function='RMSE', verbose=100, l2_leaf_reg=1)

    # early stopping setup
    early_stopping_rounds = 10

    # train the CatBoost model and record the loss every 5 epochs
    catboost_model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=early_stopping_rounds, verbose=5)

    # CatBoost prediction
    y_pred_cat = catboost_model.predict(X_test)

    # calculate CatBoost mean squared error
    mse_cat = mean_squared_error(y_test, y_pred_cat)
    print(f'CatBoost Test MSE: {mse_cat:.4f}')

    # save the CatBoost model
    catboost_model.save_model(f"models/catboost_model.json", format="json")