# separate features and target variable
features = scaled_df.drop(columns=['WFIR_EALB'])  # 将'target_column'替换为实际的目标列名
targets = scaled_df['WFIR_EALB']  # 将'target_column'替换为实际的目标列名

feature_scaler = StandardScaler()
target_scaler = StandardScaler()

features = feature_scaler.fit_transform(features)
targets = target_scaler.fit_transform(targets.values.reshape(-1, 1)).flatten()

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

r2_cat = r2_score(y_test, y_pred_cat)
print(f"R^2 Score: {r2}")

# plot predicted values against ground-truth (same format with Yan)
plt.figure(figsize=(4,3.8), dpi=200)
plt.scatter(y_test, y_pred_cat, s=25, c='#1F77B4', marker='o', alpha=0.3)
plt.plot(y_test, y_test, linestyle=':', linewidth=0.8, color='red')
plt.xlim([0, 21])
plt.ylim([0, 21])
plt.xticks(np.arange(0, 21, 5))
plt.yticks(np.arange(0, 21, 5))
plt.xlabel('True values', fontsize=11)
plt.ylabel('Predicted values', fontsize=11)
plt.title('CatBoost regression', fontsize=12)
plt.text(1, 19, f'MSE: {mse_cat:.4f}', fontsize=10)
plt.text(1, 18, f'R-squared: {r2_cat:.4f}', fontsize=10)

# plot the feature importance figure
feature_importance = catboost_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(5,7), dpi=200)
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx], fontsize=8)
plt.xlabel('Feature Importance', fontsize=11)
plt.title('CatBoost regression', fontsize=12)
plt.savefig('CatBoost_FI.png')
plt.show()
plt.close()

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt

# Define a range of hyperparameter values to search through
depth_values = [4, 8, 16]
learning_rate_values = [0.2, 0.05, 0.5]
l2_leaf_reg_values = [1, 2, 3]
best_score = 0 # Initialize the best score
best_params = {} # Initialize the best hyperparameters

# Define cross-validation settings
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize a list to store tuning progress
tuning_progress = []

# Perform hyperparameter tuning with cross-validation
for iterations in iterations_values:
	for depth in depth_values:
		for learning_rate in learning_rate_values:
            for l2_leaf_reg in l2_leaf_reg_values:
			# Create a CatBoost model with the current hyperparameters
			catboost_mode = CatBoostRegressor(iterations=iterations, depth=depth,
									learning_rate=learning_rate, loss_function='RMSE', verbose=100, l2_leaf_reg=l2_leaf_re)
			# Perform cross-validation and get the mean F1 score
			f1_scores = []
			for train_index, val_index in cv.split(X, y):
				X_train, X_val = X.iloc[train_index], X.iloc[val_index]
				y_train, y_val = y.iloc[train_index], y.iloc[val_index]
				model.fit(X_train, y_train)
				y_pred = model.predict(X_val)
				f1 = f1_score(y_val, y_pred)
				f1_scores.append(f1)

			mean_f1 = sum(f1_scores) / len(f1_scores)

			# Update the best hyperparameters if a better score is found
			if mean_f1 > best_score:
				best_score = mean_f1
				best_params = {
					'iterations': iterations,
					'depth': depth,
					'learning_rate': learning_rate
				}

			# Append the progress to the list
			tuning_progress.append({
				'Iterations': iterations,
				'Depth': depth,
				'Learning Rate': learning_rate,
				'F1 Score': mean_f1
			})
