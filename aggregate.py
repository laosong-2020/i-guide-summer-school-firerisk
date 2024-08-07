'''
Author: Zhenlei Song
Email: songzl@tamu.edu
Purpose: This project is for the I-GUIDE Summer School 2024.

This source code file defines functions 
to add columns to a GeoDataFrame based on the values of reprojected raster files.
'''

import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np
from rasterio.mask import mask

from config import nri_columns, feature_dict_list, DATASET_DIR

def add_column_by_feature(gdf, feature_dict):
    raster = rasterio.open(feature_dict["reprojected_file_name"])
    print(f"Fetching values for {feature_dict['feature_name']}...")
    value_list = []
    for _, row in gdf.iterrows():
        geom = [row['geometry'].__geo_interface__]
        out_image, out_transform = mask(raster, geom, crop=True)
        masked = np.ma.masked_array(out_image, out_image == raster.nodata)
        if feature_dict["method"] == "sum":
            sum_value = masked.sum()
            # row[feature_name] = sum_value
            value_list.append(sum_value)
        elif feature_dict["method"] == "mean":
            mean_value = masked.mean()
            # row[feature_name] = mean_value
            value_list.append(mean_value)

    raster.close()
    print(f"Adding {feature_dict['feature_name']} to the GeoDataFrame...")
    gdf.insert(2, feature_dict["feature_name"], value_list)

if __name__ == "__main__":
    # Load the census tract shapefile
    census_tracts_gdf = gpd.read_file(f"{DATASET_DIR}/tl_2023_48_tract/tl_2023_48_tract.shp")
    census_tracts_columns = ['GEOID', 'geometry']
    census_tracts_gdf = census_tracts_gdf[census_tracts_columns]
    
    # load NRI data
    nri_df = pd.read_csv(f"{DATASET_DIR}/NRI/NRI_Table_CensusTracts_Texas.csv")
    nri_df = nri_df[nri_columns]

    # Initialize a list to store the results
    results = []

    for _, tract in census_tracts_gdf.iterrows():
        # Extract the geometry in GeoJSON format
        geom = [tract['geometry'].__geo_interface__]

        results.append({
            'GeoId': tract['GEOID'],  # Assuming 'GEOID' is the identifier for the census tract
            'geometry': tract['geometry']  # Add the geometry to the result
        })

    gdf = gpd.GeoDataFrame(
        results,
        geometry='geometry',
        crs=census_tracts_gdf.crs  # Use the same CRS as the census tracts
    )
    gdf['GeoId'] = gdf['GeoId'].astype("int64")
    # merge NRI data with census tract data
    merged_gdf = gdf.merge(
        nri_df, left_on='GeoId', right_on='TRACTFIPS', how='left')
    
    # clean null rows
    merged_gdf = merged_gdf.dropna(subset=nri_columns)

    for feature_dict in feature_dict_list:
        add_column_by_feature(merged_gdf, feature_dict)
        # add_column_by_feature(gdf, feature_dict["reprojected_file_name"], feature_dict["feature_name"])

    # change columns from string to float

    merged_gdf['Min-Temperature'] = merged_gdf['Min-Temperature'].apply(lambda x: np.nan if np.ma.is_masked(x) else x).astype("float64")
    merged_gdf['Max-Temperature'] = merged_gdf['Max-Temperature'].apply(lambda x: np.nan if np.ma.is_masked(x) else x).astype("float64")
    merged_gdf['Mean-Temperature'] = merged_gdf['Mean-Temperature'].apply(lambda x: np.nan if np.ma.is_masked(x) else x).astype("float64")

    merged_gdf = merged_gdf.convert_dtypes()
    print(merged_gdf.dtypes)

    merged_gdf.to_file(f"{DATASET_DIR}/results.geojson", driver='GeoJSON')