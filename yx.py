#%% Data Cleaning - imputer¶
# load the original all data
df = pd.read_csv(f"/home/jovyan/Team5/datasets/processed/data_dropgeo.csv")
df = df.drop(columns=["WFIR_HLRR","WFIR_HLRB"])
# fill in nan values
imputer = SimpleImputer(strategy='constant', fill_value=0.0)
# fit and transform the data
df_im = imputer.fit_transform(df)

# standardize dataset
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df_im)
# convert the result back to dataframe
df_std = pd.DataFrame(scaled_df, index=df.index, columns=df.columns)

#%% XGBoost Regression
# define a function to separate the features and target variables
def create_train_test_sets(scaled_df):

    # Split the data into training and testing sets
    label_column = "WFIR_EALB"
    X = scaled_df.drop(columns=[label_column])
    y = scaled_df[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = create_train_test_sets(df_std)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#%% create XGBoost regression model
# add early stopping setup
xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10000, learning_rate=0.05,\
                                 early_stopping_rounds=10, eval_metric="rmse", reg_lambda=1)

eval_set = [(X_train, y_train), (X_test, y_test)]
#eval_set = [(X_test, y_test)]

# train the XGBoost model and record the loss every 5 epochs
xgboost_model.fit(X_train, y_train, eval_set=eval_set, verbose=5)

# XGBoost prediction
y_pred_xgb = xgboost_model.predict(X_test)

# calculate XGBoost mean squared error
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f'XGBoost Test MSE: {mse_xgb:.4f}')
print(f'XGBoost Test R-squared: {r2_xgb:.4f}')

#%% Hyperparameter tuning for XGBoost regression
param_grid = {
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
    'max_depth': [4, 6, 8, 10],
    
}

#%% Plot figures
# plot predicted values against ground-truth 
plt.figure(figsize=(4,3.8), dpi=200)
plt.scatter(y_test, y_pred_xgb, s=25, c='#1f77b4', marker='o', alpha=0.3)
plt.plot(y_test, y_test, linestyle=':', linewidth=0.8, color='red')
plt.xlim([0, 21])
plt.ylim([0, 21])
plt.xticks(np.arange(0, 21, 5))
plt.yticks(np.arange(0, 21, 5))
plt.xlabel('True values', fontsize=11)
plt.ylabel('Predicted values', fontsize=11)
plt.title('XGBoost regression', fontsize=12)

plt.text(1, 19, f'MSE: {mse_xgb:.4f}', fontsize=10)
plt.text(1, 18, f'R-squared: {r2_xgb:.4f}', fontsize=10)
#plt.show()
plt.savefig('XGBoost.png')
plt.close()

# plot the feature importance figure
feature_importance = xgboost_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(5,7), dpi=200)
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx], fontsize=8)
plt.xlabel('Feature Importance', fontsize=11)
plt.title('XGBoost regression', fontsize=12)
#plt.savefig('XGBoost_FI.png')
plt.show()
plt.close()
