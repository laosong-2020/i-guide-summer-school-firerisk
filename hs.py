# CORRELATION MATRIX

import os

import rasterio 
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point


# Path to the GeoJSON file
geojson_file = "C:\\Users\\harma\\OneDrive\\Desktop\\i-guide-summer-school-firerisk\\datasets\\results.geojson"

# Read the GeoJSON file
gdf = gpd.read_file(geojson_file)

# Print the GeoDataFrame
print(gdf)

# Plot the GeoDataFrame
gdf.plot()

import seaborn as sns
import matplotlib.pyplot as plt

# Drop geometry column to compute correlation only on attribute data
gdf_attributes = gdf.drop(columns='geometry')

# Select only numeric columns
numeric_columns = gdf_attributes.select_dtypes(include=['number']).columns
gdf_numeric = gdf_attributes[numeric_columns]

# Compute the correlation matrix
correlation_matrix = gdf_numeric.corr()

# Print the correlation matrix
#print("Correlation Matrix:")
#print(correlation_matrix)

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(15, 12))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            annot=True, fmt='.2f', annot_kws={"size": 7}, square=True, linewidths=0.5, cbar_kws={"shrink": 0.75})

# Add title and adjust layout
plt.title('Correlation Matrix', fontsize=18)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()  # Adjust the padding between and around subplots
plt.show()

#--------------------------------------------------------------------------------------------

#EXTRA TREES REGRESSOR

import joblib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Path to the CSV file (CHANGE FILE PATH)
csv_file = 'datasets/processed/scaled_results.csv'

# Read the CSV file
df = pd.read_csv(csv_file)

# Assume 'target_var' is the dependent variable and the rest are predictors
target_var = 'WFIR_EALB'  

# Ensure the target variable is in the DataFrame
if target_var not in df.columns:
    raise ValueError(f"Target variable '{target_var}' not found in the DataFrame columns.")

# Convert categorical variables to dummy variables (one-hot encoding)
df_dummies = pd.get_dummies(df, drop_first=True)

# Drop columns with all missing values
df_dummies = df_dummies.dropna(axis=1, how='all')

# Separate dependent (target) and independent (predictor) variables
y = df_dummies[target_var]
X = df_dummies.drop(columns=[target_var])

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize the Extra Trees Regressor
et = ExtraTreesRegressor(n_estimators=100, random_state=42)

# Fit the model on the training data
et.fit(X_train, y_train)

# Save the model
#joblib.dump(et, 'extra_trees_regressor_model.joblib')

# Predict on the test data
y_pred = et.predict(X_test)

# Save the results to a CSV file
#results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#results.to_csv('extra_trees_predictions.csv', index=False)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Scatter plot of actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.show()

# Feature importance
feature_importances = pd.Series(et.feature_importances_, index=X.columns).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar')
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

#--------------------------------------------------------------------------------------------

# GRADIENT BOOSTING REGRESSOR

import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Path to the CSV file (CHANGE FILE PATH)
csv_file = 'datasets/processed/scaled_results.csv'

# Read the CSV file
df = pd.read_csv(csv_file)

# Assume 'target_var' is the dependent variable and the rest are predictors
target_var = 'WFIR_EALB'  

# Ensure the target variable is in the DataFrame
if target_var not in df.columns:
    raise ValueError(f"Target variable '{target_var}' not found in the DataFrame columns.")

# Convert categorical variables to dummy variables (one-hot encoding)
df_dummies = pd.get_dummies(df, drop_first=True)

# Drop columns with all missing values
df_dummies = df_dummies.dropna(axis=1, how='all')

# Separate dependent (target) and independent (predictor) variables
y = df_dummies[target_var]
X = df_dummies.drop(columns=[target_var])

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Fit the model on the training data
gbr.fit(X_train, y_train)

# Save the model
#joblib.dump(gbr, 'gradient_boosting_regressor_model.joblib')

# Predict on the test data
y_pred = gbr.predict(X_test)

# Save the results to a CSV file
#results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#results.to_csv('gradient_boosting_predictions.csv', index=False)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Scatter plot of actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.show()

# Feature importance
feature_importances = pd.Series(gbr.feature_importances_, index=X.columns).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar')
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

#--------------------------------------------------------------------------------------------

# BAGGING REGRESSOR

import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Path to the CSV file (CHANGE FILE PATH)
csv_file = 'datasets/processed/scaled_results.csv'

# Read the CSV file
df = pd.read_csv(csv_file)

# Assume 'target_var' is the dependent variable and the rest are predictors
target_var = 'WFIR_EALB'  

# Ensure the target variable is in the DataFrame
if target_var not in df.columns:
    raise ValueError(f"Target variable '{target_var}' not found in the DataFrame columns.")

# Convert categorical variables to dummy variables (one-hot encoding)
df_dummies = pd.get_dummies(df, drop_first=True)

# Drop columns with all missing values
df_dummies = df_dummies.dropna(axis=1, how='all')

# Separate dependent (target) and independent (predictor) variables
y = df_dummies[target_var]
X = df_dummies.drop(columns=[target_var])

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize the Bagging Regressor
bagging_regressor = BaggingRegressor(n_estimators=100, random_state=42)

# Fit the model on the training data
bagging_regressor.fit(X_train, y_train)

# Save the model
#joblib.dump(bagging_regressor, 'bagging_regressor_model.joblib')

# Predict on the test data
y_pred = bagging_regressor.predict(X_test)

# Save the results to a CSV file
#results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#results.to_csv('bagging_predictions.csv', index=False)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Scatter plot of actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.show()

# Feature importance (if base estimator supports it, e.g., decision tree)
try:
    feature_importances = pd.Series(bagging_regressor.estimators_[0].feature_importances_, index=X.columns).sort_values(ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    feature_importances.plot(kind='bar')
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.show()
except AttributeError:
    print("The base estimator does not support feature importances.")

#--------------------------------------------------------------------------------------------

#RANDOM FOREST REGRESSOR

import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Path to the CSV file (CHANGE FILE PATH)
csv_file = 'datasets/processed/scaled_results.csv'

# Read the CSV file
df = pd.read_csv(csv_file)

# Assume 'target_var' is the dependent variable and the rest are predictors
target_var = 'WFIR_EALB'  

# Ensure the target variable is in the DataFrame
if target_var not in df.columns:
    raise ValueError(f"Target variable '{target_var}' not found in the DataFrame columns.")

# Convert categorical variables to dummy variables (one-hot encoding)
df_dummies = pd.get_dummies(df, drop_first=True)

# Drop columns with all missing values
df_dummies = df_dummies.dropna(axis=1, how='all')

# Separate dependent (target) and independent (predictor) variables
y = df_dummies[target_var]
X = df_dummies.drop(columns=[target_var])

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model on the training data
rf.fit(X_train, y_train)

# Save the model
#joblib.dump(rf, 'random_forest_regressor_model.joblib')

# Predict on the test data
y_pred = rf.predict(X_test)

# Save the results to a CSV file
#results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#results.to_csv('random_forest_predictions.csv', index=False)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Scatter plot of actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.show()

# Feature importance
feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar')
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

#--------------------------------------------------------------------------------------------

#MLP REGRESSOR

import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint

# Path to the CSV file (CHANGE FILE PATH)
csv_file = 'datasets/processed/scaled_results.csv'

# Read the CSV file
df = pd.read_csv(csv_file)

# Assume 'target_var' is the dependent variable and the rest are predictors
target_var = 'WFIR_EALB'  

# Ensure the target variable is in the DataFrame
if target_var not in df.columns:
    raise ValueError(f"Target variable '{target_var}' not found in the DataFrame columns.")

# Convert categorical variables to dummy variables (one-hot encoding)
df_dummies = pd.get_dummies(df, drop_first=True)

# Drop columns with all missing values
df_dummies = df_dummies.dropna(axis=1, how='all')

# Separate dependent (target) and independent (predictor) variables
y = df_dummies[target_var]
X = df_dummies.drop(columns=[target_var])

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Define the parameter distribution
param_dist = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,), (150,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': uniform(0.0001, 0.05),
    'learning_rate': ['constant','adaptive'],
    'max_iter': randint(200, 600)
}

# Initialize the MLP Regressor
mlp_regressor = MLPRegressor(random_state=42)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=mlp_regressor, param_distributions=param_dist, n_iter=100, n_jobs=-1, cv=5, scoring='r2', random_state=42, verbose=2)

# Fit the RandomizedSearchCV on the training data
random_search.fit(X_train, y_train)

# Get the best parameters
best_params = random_search.best_params_
print(f"Best parameters found: {best_params}")

# Fit the model with the best parameters on the entire training data
best_mlp_regressor = random_search.best_estimator_
best_mlp_regressor.fit(X_train, y_train)

# Save the best model
#joblib.dump(best_mlp_regressor, 'best_mlp_regressor_model.joblib')

# Predict on the test data
y_pred = best_mlp_regressor.predict(X_test)

# Save the results to a CSV file
#results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#results.to_csv('best_mlp_predictions.csv', index=False)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Scatter plot of actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.show()

# Permutation feature importance
perm_importance = permutation_importance(best_mlp_regressor, X_test, y_test, n_repeats=10, random_state=42)

# Create a DataFrame for feature importances
feature_importances = pd.DataFrame(perm_importance.importances_mean, index=X.columns, columns=['Importance']).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar', legend=False)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()