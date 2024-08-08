from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import geopandas as gpd
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sns.set_theme(style="white")

    gdf = gpd.read_file("datasets/results.geojson")
    df = pd.DataFrame(gdf.drop(columns=['geometry']))

    df_year = pd.read_csv(f"datasets/yearbuilt_housingcount.csv")
    df_year.rename(columns={'GEOID': 'GeoId'}, inplace=True)
    df = pd.merge(df, df_year, on='GeoId')
    df = df.drop(columns=['TRACTFIPS', 'WFIR_HLRB'])

    # Finalize df and gdf
    gdf = gpd.GeoDataFrame(df, geometry=gdf['geometry'], crs=gdf.crs)
    df = df.drop(columns=['GeoId'])
    # Create LabelEncoder object
    encoder = LabelEncoder()
    scaler = StandardScaler()

    df['WFIR_HLRR'] = encoder.fit_transform(df['WFIR_HLRR'])
    columns = df.columns
    scaler.fit(df)
    df = pd.DataFrame(scaler.transform(df), columns=columns)

    corr = df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    f.savefig("results/correlation.png")
    plt.close(f)
    X,y = shuffle(df.drop(columns='WFIR_EALB'), df['WFIR_EALB'], random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None )

    models,predictions = reg.fit(X_train, X_test, y_train, y_test)

    print(models)