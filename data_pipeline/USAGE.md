# Data Pipeline Usage Guide

This guide explains how to use the SAIL 2025 data pipeline for crowd prediction.

## Overview

The data pipeline processes and merges multiple data sources to create a unified dataset for machine learning models. It handles:

- **Sensor data**: Crowd flow measurements (3-minute intervals)
- **Weather data**: KNMI weather observations (resampled from 10-min to 3-min)
- **Vessel data**: Position tracking for large ships (>100m)
- **Sensor locations**: Geographic coordinates and metadata

## Quick Start

### Option 1: Run as Python Script

```bash
cd data_pipeline
python data_pipeline.py
```

This will:
1. Load all data files from the `data_pipeline/` folder
2. Process and merge the data
3. Save the result as `processed_data.csv` in the project root
4. Display detailed progress and summary statistics

### Option 2: Use in Jupyter Notebook

Copy the code from `data_pipeline_notebook_cells.txt` into your notebook. Each cell is clearly labeled and documented.

**Recommended structure:**
- Section 4 of `project_template.ipynb`: Data Pipeline (13 cells)
- Section 5: Ridge Regression Model
- Section 6: XGBoost Model
- Section 7: Model Comparison

## Data Pipeline Steps

### Step 1: Load Raw Data
- Sensor measurements (3-min intervals)
- Weather observations (10-min intervals)
- Vessel position tracking
- Sensor location metadata

### Step 2: Process Sensor Locations
- Parse latitude/longitude coordinates
- Fix decimal notation (comma → dot)
- Create effective width lookup dictionary

### Step 3: Normalize Sensor Values
- Divide raw counts by effective width
- Convert to standardized flow: (people/meter)/minute
- Ensures fair comparison across sensors

### Step 4: Process Weather Data
- Fix datetime format (hour 24 → 00)
- Convert to UTC datetime
- Resample from 10-minute to 3-minute intervals

### Step 5: Process Vessel Data
- Convert to UTC and floor to 3-minute intervals
- Aggregate positions per vessel per interval
- Filter to large vessels only (>100m)

### Step 6: Merge Data Sources
- Inner join: sensors + vessels (on timestamp)
- Left join: add weather data
- Creates combined dataset with all features

### Step 7: Calculate Spatial Features
- **Haversine distance**: Accurate great-circle distances
- **Distance features**: Vessel-to-sensor distances for all sensors
- **Distance cap**: Maximum influence at 1000 meters
- **Weighted aggregation**: Inverse-distance weighting (closer = more influence)
- **Vessel weighting**: Larger vessels have more influence

## Output Dataset

### Features (Input Variables)

**Weather Features (3):**
- `Temperature_°C`: Air temperature
- `Humidity_%`: Relative humidity
- `Rain_mm`: Precipitation amount

**Vessel Features (1 + N sensors):**
- `length_weighted`: Weighted average vessel length
- `dist_[SENSOR-ID]_weighted`: Weighted distance to each sensor

**Total:** Approximately 50-60 features depending on number of sensors

### Targets (Output Variables)

**Sensor Measurements (74 directional measurements):**
- Format: `SENSOR-ID_ANGLE`
- Example: `CMSA-GAKH-01_0`, `CMSA-GAKH-01_180`
- Unit: (people/meter)/minute (normalized flow)

### Train/Test Split

- **Training**: August 20-23, 2025 (~1,900 timestamps)
- **Test**: August 24, 2025 (~480 timestamps)
- **Split timestamp**: 2025-08-24 00:00:00+02:00

## Data Files Required

Ensure these files exist in the `data_pipeline/` folder:

```
data_pipeline/
├── sensordata_SAIL2025.csv
├── SAIL_Amsterdam_10min_Weather_2025-08-20_to_2025-08-24_FIXED.csv
├── vessels_data.parquet
└── tomtom_data.parquet (optional, for future use)
```

Also required in project root:
```
sensor-location.xlsx - Sheet1.csv
```

## Configuration

Edit these parameters in `data_pipeline.py` if needed:

```python
# Train/Test split timestamp
SPLIT_TIME = pd.Timestamp("2025-08-24 00:00:00+02:00")

# Vessel filtering: only large vessels (>100m)
MIN_VESSEL_LENGTH = 10000  # in cm

# Distance threshold for vessel influence
MAX_DISTANCE = 1000  # meters
```

## Machine Learning Integration

After running the pipeline, use the processed data with your models:

```python
# Load processed data
import pandas as pd
agg_df = pd.read_csv("processed_data.csv", parse_dates=["timestamp"])

# Define features and targets
feature_cols = [c for c in agg_df.columns 
                if c not in target_cols and c != "timestamp"]
target_cols = [c for c in agg_df.columns 
               if c.startswith(("CMSA-", "GACM-", "GASA-", "GVCV-"))]

# Split train/test
SPLIT_TIME = pd.Timestamp("2025-08-24 00:00:00+02:00")
train_data = agg_df[agg_df["timestamp"] < SPLIT_TIME]
test_data = agg_df[agg_df["timestamp"] >= SPLIT_TIME]

# Prepare matrices
X_train = train_data[feature_cols].values
X_test = test_data[feature_cols].values
y_train = train_data[target_cols].values
y_test = test_data[target_cols].values

# Train your models...
```

## Feature Engineering Details

### Haversine Distance Formula

The pipeline uses the Haversine formula for accurate distance calculation on Earth's curved surface:

```python
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))
```

### Inverse-Distance Weighting

Vessel influence decreases with distance:

```python
# Weight = 1 / (distance + 1)
# Closer vessels have higher weights
weights = 1 / (distances + 1)
weighted_average = np.average(values, weights=weights)
```

### Why This Works

1. **Spatial relevance**: Closer vessels attract more spectators
2. **Size matters**: Larger vessels are more impressive
3. **Multiple vessels**: Aggregation captures combined influence
4. **Distance cap**: No influence beyond 1000m (reasonable walking distance)

## Performance Tips

- The pipeline processes ~2,400 timestamps
- Spatial feature calculation takes 2-5 minutes
- Use `low_memory=False` for large CSV files
- Parquet files (vessels) are faster than CSV

## Troubleshooting

### "File not found" error
- Check that all data files are in the `data_pipeline/` folder
- Verify `sensor-location.xlsx - Sheet1.csv` is in project root

### "Parsing error" in weather data
- Ensure DateTime column has format: "YYYYMMDD HH:MM"
- Check for hour "24" (should be replaced with "00")

### "Empty dataset" after merge
- Check datetime formats are consistent (all UTC)
- Verify timestamp ranges overlap between datasets

### Memory issues
- Process data in chunks if needed
- Use `dtype` parameter in `pd.read_csv()` to reduce memory

## Next Steps

After completing the data pipeline:

1. **Train Ridge Regression Model**
   - Linear baseline model
   - Fast training
   - Interpretable coefficients

2. **Train XGBoost Model**
   - Non-linear model
   - Better performance
   - Feature importance analysis

3. **Compare Models**
   - RMSE, MAE metrics
   - Prediction vs actual plots
   - Feature importance

4. **Generate Predictions**
   - 5-15 minute forecasts
   - Confidence intervals
   - Real-time monitoring dashboard

## Support

For issues or questions:
- Check the README.md in project root
- Review code comments in `data_pipeline.py`
- Consult `data_descriptions.md` for data source details

## Citation

This data pipeline was developed for the TIL6022 course project analyzing crowd behavior during SAIL Amsterdam 2025.
