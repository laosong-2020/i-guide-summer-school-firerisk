'''
Author: Zhenlei Song
Email: songzl@tamu.edu
Purpose: This project is for the I-GUIDE Summer School 2024.

This source code file contains the reproject function to reproject a raster file to a target CRS.
And save the reprojected raster file to the output path.
'''

import rasterio 
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd 

from config import feature_dict_list, BASE_DIR, DATASET_DIR

def reproject_tif(target_crs, source_tif_path, output_tif_path):
    with rasterio.open(source_tif_path) as src:
        transform, width, height = calculate_default_transform(
        src.crs, target_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_tif_path, 'w', **kwargs) as dst:
            print(f"Reprojecting {source_tif_path} to {output_tif_path}...")
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest)



if __name__ == "__main__":
    # Load the census tract shapefile
    census_tracts = gpd.read_file(f"{DATASET_DIR}/tl_2023_48_tract/tl_2023_48_tract.shp")
    target_crs = census_tracts.crs

    for feature_dict in feature_dict_list:
        reproject_tif(target_crs, feature_dict["file_name"], feature_dict["reprojected_file_name"])
