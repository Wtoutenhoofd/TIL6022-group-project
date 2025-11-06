"""
SAIL 2025 Crowd Prediction - Data Pipeline
===========================================

This script processes and merges all data sources for the SAIL 2025 crowd prediction project.
It creates a unified dataset ready for machine learning models.

Data Sources:
1. Crowd sensor data (3-minute intervals)
2. Weather data from KNMI (10-minute intervals, resampled to 3-min)
3. Vessel position tracking data
4. Sensor location metadata

Output:
- Unified dataset with all features aligned to 3-minute timestamps
- Features include: weather, vessel proximity, weighted distances, and sensor measurements
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(__file__).parent  # data_pipeline folder
OUTPUT_FILE = DATA_DIR.parent / "processed_data.csv"

# File paths
PATH_SENSORS = DATA_DIR / "sensordata_SAIL2025.csv"
PATH_WEATHER = DATA_DIR / "SAIL_Amsterdam_10min_Weather_2025-08-20_to_2025-08-24_FIXED.csv"
PATH_VESSELS = DATA_DIR / "vessels_data.parquet"
PATH_SENSOR_LOC = DATA_DIR.parent / "sensor-location.xlsx - Sheet1.csv"

# Train/Test split timestamp
SPLIT_TIME = pd.Timestamp("2025-08-24 00:00:00+02:00")

# Vessel filtering: only large vessels (>100m)
MIN_VESSEL_LENGTH = 10000  # in cm (100m)

# Distance threshold for vessel influence
MAX_DISTANCE = 1000  # meters

print("=" * 70)
print("SAIL 2025 CROWD PREDICTION - DATA PIPELINE")
print("=" * 70)

# ============================================================================
# STEP 1: LOAD RAW DATA
# ============================================================================

print("\n[STEP 1/7] Loading raw data files...")

# Load sensor data
print(f"  • Loading sensor measurements: {PATH_SENSORS.name}")
sensors = pd.read_csv(PATH_SENSORS, parse_dates=["timestamp"])
print(f"    → Shape: {sensors.shape}, Columns: {len(sensors.columns)}")

# Load weather data
print(f"  • Loading weather data: {PATH_WEATHER.name}")
weather = pd.read_csv(PATH_WEATHER)
print(f"    → Shape: {weather.shape}")

# Load vessel data
print(f"  • Loading vessel positions: {PATH_VESSELS.name}")
vessels = pd.read_parquet(PATH_VESSELS)
vessels = vessels.rename(columns={"upload-timestamp": "timestamp"})
if "stale_since" in vessels.columns:
    vessels = vessels.drop(columns=["stale_since"])
print(f"    → Shape: {vessels.shape}")

# Load sensor locations
print(f"  • Loading sensor locations: {PATH_SENSOR_LOC.name}")
sensors_location = pd.read_csv(PATH_SENSOR_LOC)
sensors_location = sensors_location.rename(columns={"Objectnummer": "sensor_id"})
print(f"    → Shape: {sensors_location.shape}")

print("  ✓ All data files loaded successfully")

# ============================================================================
# STEP 2: PREPROCESS SENSOR LOCATION DATA
# ============================================================================

print("\n[STEP 2/7] Processing sensor location metadata...")

# Fix decimal notation (comma → dot)
sensors_location["Effectieve breedte"] = (
    sensors_location["Effectieve breedte"]
    .astype(str)
    .str.replace(",", ".")
    .astype(float)
)

# Parse lat/lon coordinates
sensors_location[["lat", "lon"]] = (
    sensors_location["Lat/Long"]
    .str.replace(" ", "")
    .str.split(",", expand=True)
    .astype(float)
)

print(f"  • Parsed {len(sensors_location)} sensor locations")
print(f"  ✓ Sensor metadata processed")

# ============================================================================
# STEP 3: NORMALIZE SENSOR VALUES BY EFFECTIVE WIDTH
# ============================================================================

print("\n[STEP 3/7] Normalizing sensor measurements by effective width...")

# Sensor prefixes to identify measurement columns
sensor_prefixes = ("CMSA-", "GACM-", "GASA-", "GVCV-")

# Create lookup dictionary: sensor_id → effective_width
width_lookup = sensors_location.set_index("sensor_id")["Effectieve breedte"].to_dict()

# Normalize each sensor column
normalized_count = 0
for col in sensors.columns:
    if "_" in col:  # Sensor columns have format: SENSOR-ID_ANGLE
        sensor_id = col.split("_")[0]
        if sensor_id in width_lookup:
            sensors[col] = sensors[col] / width_lookup[sensor_id]
            normalized_count += 1

print(f"  • Normalized {normalized_count} sensor measurement columns")
print(f"  ✓ Sensor normalization complete")

# ============================================================================
# STEP 4: PREPROCESS WEATHER DATA
# ============================================================================

print("\n[STEP 4/7] Processing weather data...")

# Fix datetime format (replace hour 24 with 00)
weather["DateTime"] = weather["DateTime"].str.replace(" 24:", " 00:", regex=False)

# Convert to UTC datetime
weather["DateTime"] = pd.to_datetime(weather["DateTime"], format="%Y%m%d %H:%M")
weather = weather.set_index("DateTime")

# Resample from 10-minute to 3-minute intervals
weather_3min = weather.resample("3min").nearest()
weather_3min = weather_3min.reset_index()

print(f"  • Resampled weather data from 10-min to 3-min intervals")
print(f"  • Weather features: {', '.join(weather_3min.columns[1:])}")
print(f"  ✓ Weather processing complete")

# ============================================================================
# STEP 5: PREPROCESS VESSEL DATA
# ============================================================================

print("\n[STEP 5/7] Processing vessel position data...")

# Convert to UTC datetime and floor to 3-minute intervals
vessels["timestamp"] = pd.to_datetime(vessels["timestamp"], utc=True, errors="coerce")
vessels["timestamp"] = vessels["timestamp"].dt.floor("3min")

# Aggregate vessel positions per 3-minute interval
vessels = (
    vessels.groupby(["timestamp", "imo-number"], as_index=False)
    .agg({
        "lat": "mean",
        "lon": "mean",
        "length": "first"
    })
    .dropna(subset=["timestamp", "imo-number", "lat", "lon", "length"])
)

# Filter: only large vessels (>100m)
vessels = vessels[vessels["length"] > MIN_VESSEL_LENGTH]

print(f"  • Aggregated vessel positions to 3-min intervals")
print(f"  • Filtered to vessels > {MIN_VESSEL_LENGTH/100:.0f}m: {vessels['imo-number'].nunique()} unique vessels")
print(f"  ✓ Vessel processing complete")

# ============================================================================
# STEP 6: MERGE DATA SOURCES
# ============================================================================

print("\n[STEP 6/7] Merging all data sources...")

# Ensure datetime compatibility
vessels["timestamp"] = pd.to_datetime(vessels["timestamp"], utc=True)
sensors["timestamp"] = pd.to_datetime(sensors["timestamp"], utc=True)
weather_3min["DateTime"] = pd.to_datetime(weather_3min["DateTime"], utc=True)

# Merge sensors + vessels
combined = sensors.merge(vessels, on="timestamp", how="inner")
print(f"  • Merged sensors + vessels: {combined.shape}")

# Merge with weather
combined = combined.merge(
    weather_3min.rename(columns={"DateTime": "timestamp"})[
        ["timestamp", "Temperature_°C", "Humidity_%", "Rain_mm"]
    ],
    on="timestamp",
    how="left"
)
print(f"  • Added weather data: {combined.shape}")
print(f"  ✓ Data merge complete")

# ============================================================================
# STEP 7: CALCULATE VESSEL-SENSOR DISTANCES & FEATURES
# ============================================================================

print("\n[STEP 7/7] Computing spatial features (vessel-sensor distances)...")

# Haversine formula for accurate distance on Earth's surface
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two points on Earth.
    Returns distance in meters.
    """
    R = 6371000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# Calculate distance from each vessel to each sensor
print(f"  • Computing distances for {len(sensors_location)} sensors...")
for _, sensor in sensors_location.iterrows():
    sensor_id = sensor["sensor_id"]
    s_lat = sensor["lat"]
    s_lon = sensor["lon"]
    
    dist_col = f"dist_{sensor_id}"
    combined[dist_col] = haversine(s_lat, s_lon, combined["lat"], combined["lon"])

print(f"  • Created {len(sensors_location)} distance features")

# Cap all distances at MAX_DISTANCE (no influence beyond that)
for col in [c for c in combined.columns if c.startswith("dist_")]:
    combined[col] = combined[col].clip(0, MAX_DISTANCE)

print(f"  • Applied distance cap at {MAX_DISTANCE}m")

# ============================================================================
# WEIGHTED AGGREGATION PER TIMESTAMP
# ============================================================================

print(f"  • Aggregating vessel influences per timestamp...")

agg_rows = []
for ts, group in combined.groupby("timestamp"):
    # Compute inverse-distance weights per vessel (closer = more influence)
    dist_cols = [c for c in group.columns if c.startswith("dist_")]
    weights = 1 / (group[dist_cols].clip(lower=1))  # avoid division by zero
    
    temp = {
        "timestamp": ts,
        # Weather features (same for all vessels at this timestamp)
        "Temperature_°C": group["Temperature_°C"].iloc[0],
        "Humidity_%": group["Humidity_%"].iloc[0],
        "Rain_mm": group["Rain_mm"].iloc[0],
    }
    
    # Weighted vessel length (larger vessels = more influence)
    temp["length_weighted"] = np.average(group["length"], weights=weights.mean(axis=1))
    
    # Weighted mean distances (closer vessels have more influence)
    for col in dist_cols:
        temp[f"{col}_weighted"] = np.average(group[col], weights=1 / (group[col] + 1))
    
    # Sensor target values (same for all vessels per timestamp)
    for target_col in [c for c in combined.columns if c.startswith(sensor_prefixes)]:
        temp[target_col] = group[target_col].iloc[0]
    
    agg_rows.append(temp)

agg_df = pd.DataFrame(agg_rows)
print(f"  • Aggregated to {len(agg_df)} timestamps")

# ============================================================================
# FINAL CLEANUP
# ============================================================================

# Remove rows with missing values
initial_rows = len(agg_df)
agg_df = agg_df.dropna()
removed_rows = initial_rows - len(agg_df)

if removed_rows > 0:
    print(f"  • Removed {removed_rows} rows with missing values")

print(f"  ✓ Spatial feature engineering complete")

# ============================================================================
# SAVE PROCESSED DATA
# ============================================================================

print("\n" + "=" * 70)
print("DATA PIPELINE SUMMARY")
print("=" * 70)

# Identify feature and target columns
feature_cols = (
    ["Temperature_°C", "Humidity_%", "Rain_mm", "length_weighted"]
    + [c for c in agg_df.columns if c.startswith("dist_")]
)
target_cols = [c for c in agg_df.columns if c.startswith(sensor_prefixes)]

print(f"\n📊 Final Dataset Shape: {agg_df.shape}")
print(f"   • Timestamps: {len(agg_df)}")
print(f"   • Features: {len(feature_cols)}")
print(f"   • Targets (sensors): {len(target_cols)}")

# Train/Test split info
train_mask = agg_df["timestamp"] < SPLIT_TIME
test_mask = ~train_mask

print(f"\n📅 Train/Test Split:")
print(f"   • Training data: {train_mask.sum()} timestamps (before {SPLIT_TIME.date()})")
print(f"   • Test data: {test_mask.sum()} timestamps (on {SPLIT_TIME.date()})")

# Save to CSV
agg_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n💾 Processed data saved to: {OUTPUT_FILE.name}")
print(f"   Full path: {OUTPUT_FILE}")

print("\n✅ DATA PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 70)

# ============================================================================
# RETURN DATA (when imported as module)
# ============================================================================

if __name__ == "__main__":
    print("\n📝 Next steps:")
    print("   1. Load the processed data: pd.read_csv('processed_data.csv')")
    print("   2. Train machine learning models (Ridge, XGBoost)")
    print("   3. Generate predictions for crowd management")
