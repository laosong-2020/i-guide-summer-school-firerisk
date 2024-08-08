import os

import rasterio 
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.getcwd()
print(f"BASE_DIR is: {BASE_DIR}")
DATASET_DIR = f"/share/Summer-School-2024/Team5/datasets"
REPROJECT_PATH = f"{BASE_DIR}/reprojected-datasets"
OUTPUT_PATH = f"{BASE_DIR}/outputs"

# load the results
data_results = gpd.read_file(f"{OUTPUT_PATH}/results.geojson")
data_results.shape
# drop some columns
df = data_results.drop(columns=['GeoId','geometry','TRACTFIPS','WFIR_HLRR','WFIR_HLRB'])
print(df.dtypes)
# calculate the correlation matrix
corrM = df.corr()

# plot the correlation matrix with upper triangle masked
mask = np.triu(np.ones_like(corrM, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corrM, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
plt.show()
#f.savefig("correlation.png")
plt.close(f)   

# standardize the data
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
# add metadata to the standardzied dataset
scaled_dfnew = pd.DataFrame(scaled_df, index=df.index, columns=df.columns)
# calculate the correlation matrix after normalization
corrS = scaled_dfnew.corr()
mask = np.triu(np.ones_like(corrS, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corrS, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
plt.show()
plt.close(f)   

# apply PCA to the standardized data

# initialize PCA with the number of components
pca = PCA(n_components=2)

# fit and transform the data
features = scaled_dfnew.columns
principal_components = pca.fit_transform(scaled_dfnew[features[1:7]])

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1','PC2'])

# Show the explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)

