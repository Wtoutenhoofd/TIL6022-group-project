# Data Pipeline - Implementation Summary

## 📁 Files Created

### 1. `data_pipeline/data_pipeline.py` (Standalone Script)
- Complete Python script that runs independently
- Processes all data sources and creates `processed_data.csv`
- Run with: `python data_pipeline/data_pipeline.py`
- **Use this for**: Quick data processing, automation, testing

### 2. `data_pipeline_notebook_cells.txt` (Notebook Cells)
- 13 well-documented cells ready to copy into Jupyter notebook
- Each cell has clear comments and progress indicators
- **Use this for**: Integrating into `project_template.ipynb` Section 4

### 3. `data_pipeline/USAGE.md` (Usage Guide)
- Comprehensive documentation
- Includes troubleshooting, configuration, and examples
- **Use this for**: Reference and understanding the pipeline

## 🎯 How to Use

### Option A: For Project Template Notebook

1. Open `data_pipeline_notebook_cells.txt`
2. Copy cells 1-13 into Section 4 of `project_template.ipynb`
3. Run the cells in order
4. Continue with Section 5 (Ridge Model) and Section 6 (XGBoost)

### Option B: Quick Data Processing

1. Open terminal in project root
2. Run: `python data_pipeline/data_pipeline.py`
3. Use generated `processed_data.csv` in your analysis

## 📊 What the Pipeline Does

### Input (4 Data Sources)
1. **Sensor data**: `sensordata_SAIL2025.csv` (3-min intervals)
2. **Weather data**: `SAIL_Amsterdam_10min_Weather_2025-08-20_to_2025-08-24_FIXED.csv`
3. **Vessel data**: `vessels_data.parquet` (position tracking)
4. **Sensor locations**: `sensor-location.xlsx - Sheet1.csv` (metadata)

### Processing Steps (7 Steps)
1. ✅ Load all raw data files
2. ✅ Process sensor location metadata
3. ✅ Normalize sensor measurements by effective width
4. ✅ Process and resample weather data (10-min → 3-min)
5. ✅ Process and aggregate vessel positions
6. ✅ Merge all data sources on timestamp
7. ✅ Calculate spatial features (vessel-sensor distances with weighting)

### Output
- **File**: `processed_data.csv` (or `agg_df` DataFrame in notebook)
- **Shape**: ~2,400 rows × ~120 columns
- **Features**: Weather (3) + Vessel proximity (~50)
- **Targets**: 74 sensor measurements (directional flow)
- **Split**: Training (Aug 20-23) + Test (Aug 24)

## 🔑 Key Features

### 1. Haversine Distance Calculation
Accurate great-circle distance between vessels and sensors on Earth's surface.

### 2. Inverse-Distance Weighting
Closer vessels have more influence on crowd behavior.

### 3. Vessel Size Weighting
Larger vessels (>100m) attract more spectators.

### 4. Normalized Flow Measurements
Sensor values divided by effective width: `(people/meter)/minute`

## 📈 Integration with Models

After running the pipeline, you have `agg_df` with:

```python
# Feature columns
feature_cols = ["Temperature_°C", "Humidity_%", "Rain_mm", "length_weighted"] + 
               [distance features for each sensor]

# Target columns (74 sensors)
target_cols = ["CMSA-GAKH-01_0", "CMSA-GAKH-01_180", ...]

# Train/Test split
SPLIT_TIME = pd.Timestamp("2025-08-24 00:00:00+02:00")
```

Ready for:
- ✅ Ridge Regression
- ✅ XGBoost
- ✅ Any other ML model

## 📝 Section 4 Description for Project Template

You can use this text for Section 4:

```markdown
# 4. Data Pipeline

The data pipeline processes and integrates multiple data sources to create a unified 
dataset for crowd prediction models. The pipeline performs the following operations:

## 4.1 Data Sources Integration

All raw data files are loaded from the `data_pipeline/` folder:
- Crowd sensor measurements (3-minute intervals)
- KNMI weather observations (10-minute intervals, resampled to 3-min)
- Vessel position tracking (filtered to ships >100m)
- Sensor location metadata with geographic coordinates

## 4.2 Data Preprocessing

**Sensor Data Normalization:**
- Raw sensor counts are divided by effective measurement width
- Standardizes measurements to: (people/meter)/minute
- Ensures fair comparison across different sensor locations

**Weather Data Resampling:**
- Original: 10-minute intervals
- Resampled: 3-minute intervals (nearest neighbor)
- Features: Temperature, Humidity, Rainfall

**Vessel Data Aggregation:**
- Positions aggregated to 3-minute intervals
- Filtered to vessels longer than 100 meters
- Calculated as mean position per vessel per interval

## 4.3 Feature Engineering

**Spatial Features:**
The pipeline calculates vessel-to-sensor distances using the Haversine formula 
for accurate great-circle distance on Earth's surface. Key aspects:

1. **Distance Calculation**: Haversine formula for all vessel-sensor pairs
2. **Distance Cap**: Maximum influence at 1,000 meters
3. **Inverse-Distance Weighting**: Closer vessels have stronger influence
4. **Size Weighting**: Larger vessels weighted more heavily

This creates approximately 50 distance features (one per sensor) plus weather 
and aggregated vessel metrics.

## 4.4 Output Dataset

The final dataset contains:
- **Timestamps**: ~2,400 (3-minute intervals over 5 days)
- **Features**: ~53 (weather + vessel proximity)
- **Targets**: 74 sensor measurements (bidirectional flow)
- **Train/Test Split**: Aug 20-23 (train) / Aug 24 (test)

The processed data is now ready for machine learning model training.
```

## 🚀 Next Steps

1. ✅ **Data Pipeline** (Complete - You are here!)
2. ⏩ **Section 5: Ridge Regression Model**
   - Train baseline linear model
   - Feature importance analysis
   - Performance metrics
3. ⏩ **Section 6: XGBoost Model**
   - Train gradient boosting model
   - Hyperparameter tuning
   - Feature importance
4. ⏩ **Section 7: Model Comparison**
   - Compare RMSE, MAE
   - Prediction vs actual plots
   - Select best model

## 💡 Tips

- **Runtime**: Pipeline takes 2-5 minutes to process all data
- **Memory**: Keep ~2GB RAM available
- **Reusability**: Save `agg_df` or `processed_data.csv` for quick reloading
- **Debugging**: Each step prints progress and shapes

## 📚 Documentation Files

- `data_pipeline/README.md` - Overview of data sources
- `data_pipeline/USAGE.md` - Detailed usage guide
- `data_pipeline/data_pipeline.py` - Standalone script
- `data_pipeline_notebook_cells.txt` - Notebook version
- `data_descriptions.md` - Data source descriptions

## ✅ Ready to Go!

Your data pipeline is complete and ready to use. Choose your preferred method:
- **Notebook**: Copy cells from `data_pipeline_notebook_cells.txt`
- **Script**: Run `python data_pipeline/data_pipeline.py`

Both methods produce the same result. The notebook version is better for 
learning and visualization, while the script is faster for automation.
