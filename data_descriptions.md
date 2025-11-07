# Data Used in This Project

All data sets used in this project can be found in the `data_pipeline/` folder.  

The data sets used have been shared via the TIL6022 course professors and collected from various sources during the SAIL 2025 event (August 20-24, 2025).  

---

## Data Source 1 - Crowd Sensor Data
**File:** `sensordata_SAIL2025.csv`  

**Description:**  
Real-time crowd flow measurements from multiple sensors placed throughout the Amsterdam event area during SAIL 2025. The sensors measure pedestrian flow in different directions, providing bidirectional crowd movement data.

**Content:**
- **Timestamp:** Date and time of measurement (3-minute intervals)
- **Sensor measurements:** Multiple columns representing different sensors with their measurement angles (e.g., `CMSA-GAKH-01_0`, `CMSA-GAKH-01_180`)
- **Format:** Each sensor has two directional measurements (typically 180° apart)
- **Coverage:** August 20-24, 2025
- **Sensors include:** CMSA, GACM, GASA, and GVCV series sensors
- **Measurement unit:** People per meter per minute (normalized by effective width of measurement location)

**Usage in project:**  
This is the primary target variable for our prediction models. The sensor data is used to train and validate both Ridge regression and XGBoost models for short-term crowd behavior forecasting.

---

## Data Source 2 - Weather Data (KNMI)
**File:** `SAIL_Amsterdam_10min_Weather_2025-08-20_to_2025-08-24.csv` (and variations: `_FIXED.csv`, `_corrected.csv`)  

**Description:**  
Weather observations from the Royal Netherlands Meteorological Institute (KNMI) for Amsterdam during the SAIL 2025 event. Weather conditions significantly influence crowd movement patterns and visitor behavior during outdoor events.

**Content:**
- **DateTime:** Timestamp of weather observation (10-minute intervals, resampled to 3-minute)
- **Temperature_°C:** Air temperature in degrees Celsius
- **Humidity_%:** Relative humidity percentage
- **Rain_mm:** Precipitation amount in millimeters
- **Coverage:** August 20-24, 2025
- **Source:** KNMI weather station data for Amsterdam area

**Usage in project:**  
Weather variables are used as features in the machine learning models to account for environmental factors affecting crowd behavior. Temperature, humidity, and rainfall patterns help explain variations in crowd density and movement.

---

## Data Source 3 - Vessel Position Data
**Files:** `vessels_data.parquet` or `Vesselposition_data_20-24Aug2025.csv`  

**Description:**  
Real-time position tracking data for vessels participating in SAIL Amsterdam 2025. As a nautical event, the presence and movement of large vessels significantly impacts crowd distribution as spectators gather to view the ships.

**Content:**
- **timestamp/upload-timestamp:** Time of vessel position update
- **imo-number:** International Maritime Organization ship identification number
- **lat:** Latitude coordinate of vessel
- **lon:** Longitude coordinate of vessel
- **length:** Vessel length in meters
- **Coverage:** August 20-24, 2025
- **Focus:** Vessels larger than 100 meters (length > 10,000 cm in the dataset)

**Usage in project:**  
Vessel proximity to sensors is calculated using the Haversine formula. Weighted distance features are created for each sensor, where closer and larger vessels have more influence. These features help predict crowd surges near waterfront locations when impressive vessels are nearby.

---

## Data Source 4 - Sensor Location Data
**File:** `sensor-location.xlsx - Sheet1.csv`  

**Description:**  
Geographic coordinates and metadata for all crowd sensors deployed during SAIL 2025. This reference data enables spatial analysis and visualization of crowd patterns.

**Content:**
- **Objectnummer:** Unique sensor identifier (e.g., CMSA-GAKH-01, GASA-01-A1)
- **Lat/Long:** Geographic coordinates (latitude, longitude)
- **Effectieve breedte:** Effective measurement width in meters
- **Location names:** Descriptive names of sensor placement locations

**Usage in project:**  
Essential for:
1. Calculating distances between sensors and vessels
2. Creating vector maps showing crowd flow directions
3. Normalizing crowd counts by effective sensor width
4. Visualizing spatial patterns on Amsterdam city maps

---

## Data Source 5 - Traffic Data (TomTom)
**File:** `tomtom_data.parquet`  

**Description:**  
Vehicular traffic flow data from TomTom navigation services for the Amsterdam area during SAIL 2025. Traffic patterns can indicate accessibility and congestion, which indirectly affects pedestrian crowd behavior.

**Content:**
- Traffic flow metrics
- Congestion levels
- Road segment information
- **Coverage:** August 20-24, 2025

**Usage in project:**  
Provides contextual information about overall mobility patterns in the city during the event. May be used as additional features to understand how vehicular traffic restrictions or congestion affect pedestrian movement.

---

## Auxiliary Data Files
- **Predicted outputs:** `predicted_sensor_values_3min.csv`, `predicted_sensor_values_3min_xgb.csv` - Model predictions for visualization and evaluation
- **Trained models:** `xgb_models/` folder containing saved XGBoost models for each sensor and angle combination
- **GeoJSON:** Amsterdam city district boundaries loaded via URL for map visualizations

---

**Data Pipeline Overview:**  
All raw data sources are preprocessed, synchronized to 3-minute intervals, and merged into a unified dataset for model training. The pipeline includes:
1. Timestamp standardization (UTC conversion)
2. Data resampling (10-min weather → 3-min intervals)
3. Spatial feature engineering (vessel-sensor distances)
4. Sensor value normalization by effective width
5. Feature aggregation and cleaning
