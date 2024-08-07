'''
Author: Zhenlei Song
Email: songzl@tamu.edu
Purpose: This project is for the I-GUIDE Summer School 2024.

This file contains the configuration for the project.
It defines feature names, feature original source paths, reprojected paths, and methods for processing the features.
'''

import os

BASE_DIR = os.getcwd()
DATASET_DIR = f"{BASE_DIR}/datasets"

nri_columns = [
        'TRACTFIPS',
        'POPULATION',
        'BUILDVALUE',
        'AGRIVALUE',
        'WFIR_AFREQ',
        'WFIR_EXPB',
        'WFIR_EXPP',
        'WFIR_EXPPE',
        'WFIR_EXPA',
        'WFIR_EXPT',
        'WFIR_HLRB',
        'WFIR_HLRP',
        'WFIR_HLRA',
        'WFIR_HLRR',
        'WFIR_EALT',
    ]

feature_dict_list = [
    {
        "file_name": f"{DATASET_DIR}/ClimateEngineTifs/DAYMET_Precipitation_JJA.tif",
        "feature_name": "Precipitation",
        "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/Precipitation.tif",
        "method": "mean"
    },
    {
        "file_name": f"{DATASET_DIR}/ClimateEngineTifs/DAYMET_tmax_JJA.tif",
        "feature_name": "Max-Temperature",
        "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/MaxTemperature.tif",
        "method": "mean"
    },
    {
        "file_name": f"{DATASET_DIR}/ClimateEngineTifs/DAYMET_tmin_JJA.tif",
        "feature_name": "Min-Temperature",
        "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/MinTemperature.tif",
        "method": "mean"
    },
    {
        "file_name": f"{DATASET_DIR}/ClimateEngineTifs/DAYMET_tmean_JJA.tif",
        "feature_name": "Mean-Temperature",
        "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/MeanTemperature.tif",
        "method": "mean"
    },
    {
        "file_name": f"{DATASET_DIR}/ClimateEngineTifs/DAYMET_HargreavesPotentialEvapotranspiration_JJA.tif",
        "feature_name": "Potential-Evapotranspiration",
        "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/PotentialEvapotranspiration.tif",
        "method": "mean"
    },
    {
        "file_name": f"{DATASET_DIR}/TX2/BuildingCount_TX.tif",
        "feature_name": "Building-Count",
        "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/BuildingCount.tif",
        "method": "sum"
    },
    {
        "file_name": f"{DATASET_DIR}/TX2/BuildingCover_TX.tif",
        "feature_name": "Building-Cover",
        "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/BuildingCover.tif",
        "method": "mean"
    },
    {
        "file_name": f"{DATASET_DIR}/TX2/BuildingDensity_TX.tif",
        "feature_name": "Building-Density",
        "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/BuildingDensity.tif",
        "method": "mean"
    },
    {
        "file_name": f"{DATASET_DIR}/TX2/HUCount_TX.tif",
        "feature_name": "Housing-Unit-Count",
        "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/HUCount.tif",
        "method": "sum"
    },
    {
        "file_name": f"{DATASET_DIR}/TX2/HUDen_TX.tif",
        "feature_name": "Housing-Unit-Density",
        "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/HUDen.tif",
        "method": "mean"
    },
    {
        "file_name": f"{DATASET_DIR}/TX2/HUExposure_TX.tif",
        "feature_name": "Housing-Unit-Exposure",
        "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/HUExposure.tif",
        "method": "sum"
    },
    
    {
        "file_name": f"{DATASET_DIR}/TX2/PopDen_TX.tif",
        "feature_name": "Population-Density",
        "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/PopDen.tif",
        "method": "mean"
    },

    {
        "file_name": f"{DATASET_DIR}/TX/CFL_TX.tif",
        "feature_name": "Conditional-Flame-Length",
        "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/CFL.tif",
        "method": "mean"
    },
    {
        "file_name": f"{DATASET_DIR}/TX/CRPS_TX.tif",
        "feature_name": "Conditional-Risk-2-Potential-Structures",
        "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/CRPS.tif",
        "method": "mean"
    },
    {
        "file_name": f"{DATASET_DIR}/TX/FLEP4_TX.tif",
        "feature_name": "Flame-Length-Over-4-Feet",
        "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/FLEP4.tif",
        "method": "mean"
    },
    {
        "file_name": f"{DATASET_DIR}/TX/FLEP8_TX.tif",
        "feature_name": "Flame-Length-Over-8-Feet",
        "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/FLEP8.tif",
        "method": "mean"
    },
    {
        "file_name": f"{DATASET_DIR}/TX/RPS_TX.tif",
        "feature_name": "Risk-2-Potential-Structures",
        "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/RPS.tif",
        "method": "mean"
    },
    {
        "file_name": f"{DATASET_DIR}/TX/BP_TX.tif",
        "feature_name": "Burn-Probability",
        "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/BP.tif",
        "method": "mean"
    },
    {
        "file_name": f"{DATASET_DIR}/TX/Exposure_TX.tif",
        "feature_name": "Exposure-type",
        "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/Exposure.tif",
        "method": "mean"
    },
]

# useless columns
"""
{
    "file_name": f"{DATASET_DIR}/TX2/PopCount_TX.tif",
    "feature_name": "Population-Count",
    "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/PopCount.tif",
    "method": "sum"
},
{
    "file_name": f"{DATASET_DIR}/TX2/HUImpact_TX.tif",
    "feature_name": "Housing-Unit-Impact",
    "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/HUImpact.tif",
    "method": "sum"
},
{
    "file_name": f"{DATASET_DIR}/TX2/HURisk_TX.tif",
    "feature_name": "Housing-Unit-Risk",
    "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/HURisk.tif",
    "method": "mean"
},
{
    "file_name": f"{DATASET_DIR}/TX/WHP_TX.tif",
    "feature_name": "Wildfire-Hazard-Potential-index",
    "reprojected_file_name": f"{DATASET_DIR}/TX_reprojected/WHP.tif",
    "method": "mean"
},
"""