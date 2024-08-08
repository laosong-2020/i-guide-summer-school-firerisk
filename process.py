import numpy as np
import pandas as pd
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from config import DATASET_DIR, RESULTS_DIR

def pca_reduction(scaled_df, columns_to_reduce, n_components=2):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_df)
    
    return pca_result

def create_train_test_sets(scaled_df):

    # Split the data into training and testing sets
    label_column = "WFIR_EALB"
    X = scaled_df.drop(columns=[label_column])
    y = scaled_df[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = pd.read_csv(f"{DATASET_DIR}/results.csv")

    df = df.drop(
        columns=[
            "WFIR_HLRR",
            "WFIR_HLRB",
        ]
    )

    # standardize the data
    scaler = StandardScaler()
    
    scaled_df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    scaled_df.to_csv(f"{DATASET_DIR}/processed/scaled_results.csv", index=False)

    # calculate the correlation matrix after normalization
    correlation_matrix = scaled_df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    plt.figure(figsize=(15, 12))
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
    plt.savefig(f"{RESULTS_DIR}/correlation_matrix.png")
    plt.close()

    X_train, X_test, y_train, y_test = create_train_test_sets(scaled_df)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)