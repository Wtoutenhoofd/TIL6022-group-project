# Data Pipeline Folder

This folder contains all the data sources used in the SAIL 2025 crowd prediction project.

## Files Included:

1. **sensordata_SAIL2025.csv** - Crowd sensor measurements (3-minute intervals)
2. **SAIL_Amsterdam_10min_Weather_2025-08-20_to_2025-08-24_FIXED.csv** - Weather data from KNMI
3. **vessels_data.parquet** - Vessel position tracking data
4. **tomtom_data.parquet** - Traffic flow data
5. **sensor-location.xlsx - Sheet1.csv** - Geographic coordinates of sensors (copy from main folder if needed)

## Data Pipeline Process:

The data pipeline performs the following steps:

1. **Load raw data files** from this folder
2. **Timestamp standardization** - Convert all timestamps to UTC
3. **Resampling** - Standardize to 3-minute intervals
4. **Feature engineering** - Calculate vessel-sensor distances, directional angles, weighted proximity
5. **Data merging** - Combine all data sources on timestamp
6. **Normalization** - Apply effective width normalization to sensor values
7. **Output** - Unified dataset ready for machine learning models

All preprocessing scripts should reference files from this folder.
