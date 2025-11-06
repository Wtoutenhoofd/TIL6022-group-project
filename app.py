# ---------------------------------------------------------------
# SAIL CROWD DASHBOARD (Version B + Vector Map Integration)
# PART 1/2 — Imports, Loaders, Sidebar, and Tabs Setup
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from urllib.request import urlopen
import re
from datetime import time

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------

# File names EXACT as confirmed by you
LOC_FILE = "sensor-location.xlsx - Sheet1.csv"          # Sensor coordinates
PRED_FILE = "predicted_sensor_values_3min.csv"          # Predicted sensor values (with angles)
REAL_FILE = None  # We will auto-detect real sensor data file later

# Stop app if file missing
def require_file(path: str):
    try:
        open(path, "r").close()
    except:
        st.error(f"Required file not found: {path}")
        st.stop()

require_file(LOC_FILE)
require_file(PRED_FILE)

# ---------------------------------------------------------------
# DATA LOADING FUNCTIONS
# ---------------------------------------------------------------

@st.cache_data
def load_locations():
    """
    Load sensor location file and extract Objectnummer, Lat, Long.
    Handles both 'Lat/Long' in one column or separate columns.
    """
    df = pd.read_csv(LOC_FILE)
    df.columns = df.columns.str.strip()

    # Identify sensor ID column
    if "Objectnummer" in df.columns:
        sid = "Objectnummer"
    else:
        # Try fuzzy match
        sid = next((c for c in df.columns if "object" in c.lower() or "sensor" in c.lower()), None)
        if sid is None:
            st.error("Could not find a sensor ID column in location file.")
            st.stop()

    # Extract coordinates
    if "Lat/Long" in df.columns:
        parts = df["Lat/Long"].astype(str).str.split(",", expand=True)
        df["Lat"] = pd.to_numeric(parts[0], errors="coerce")
        df["Long"] = pd.to_numeric(parts[1], errors="coerce")
    else:
        # Try find separate columns
        lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
        lon_col = next((c for c in df.columns if "lon" in c.lower() or "long" in c.lower()), None)
        df["Lat"] = pd.to_numeric(df[lat_col], errors="coerce")
        df["Long"] = pd.to_numeric(df[lon_col], errors="coerce")

    df.rename(columns={sid: "Objectnummer"}, inplace=True)
    df["Objectnummer"] = df["Objectnummer"].astype(str).str.strip()

    # Remove rows missing coordinates
    return df[["Objectnummer", "Lat", "Long"]].dropna()

@st.cache_data
def load_predicted_vectors():
    """
    Load predicted data and convert to long format:
    - Detect timestamp column
    - Detect if angle is in degrees or radians
    - Handles wide formats and merges '-B' angle columns into main ones
    Returns:
        vec_long: timestamp, Objectnummer, angle_deg, value
        persons_long: timestamp, Objectnummer, Persons (sum of values per sensor)
    """
    df = pd.read_csv(PRED_FILE)
    df.columns = df.columns.str.strip()

    if df.empty:
        st.error("Predicted file is empty.")
        st.stop()

    # Identify timestamp column
    tcol = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), df.columns[0])
    df["timestamp"] = pd.to_datetime(df[tcol], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)

    # Merge "-B" columns if exist
    bcols = [c for c in df.columns if "-B_" in c]
    for bc in bcols:
        main = bc.replace("-B_", "_")
        if main in df.columns:
            df[main] = df[main].fillna(0) + df[bc].fillna(0)
            df.drop(columns=[bc], inplace=True)

    # Melt into long format
    value_cols = [c for c in df.columns if c not in ["timestamp", tcol]]
    long_df = df.melt(id_vars="timestamp", value_vars=value_cols,
                      var_name="column", value_name="value")

    # Extract Objectnummer + angle
    m = long_df["column"].str.extract(r"^(.*)_(\d+)$")
    long_df["Objectnummer"] = m[0].astype(str).str.strip()
    long_df["angle_raw"] = pd.to_numeric(m[1], errors="coerce")

    long_df.dropna(subset=["angle_raw"], inplace=True)

    # Detect radians vs degrees
    max_angle = long_df["angle_raw"].max()
    if max_angle <= 6.5:  # radians
        long_df["angle_deg"] = np.degrees(long_df["angle_raw"])
    else:
        long_df["angle_deg"] = long_df["angle_raw"]

    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce").fillna(0)

    # Compute per-sensor persons (sum of all angle values)
    persons = long_df.groupby(["timestamp", "Objectnummer"], as_index=False)["value"].sum()
    persons.rename(columns={"value": "Persons"}, inplace=True)

    return long_df, persons

# Load data
locations = load_locations()
vec_long, persons_pred = load_predicted_vectors()

# If real data exists, we will load later in Part 2

# ---------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------

st.sidebar.title("Filters")

# Available dates from predicted data
all_dates = sorted(vec_long["timestamp"].dt.date.unique())
date_sel = st.sidebar.selectbox("Select Date", all_dates)

# Time range filter
start_t = st.sidebar.time_input("Start Time", time(9, 0))
end_t = st.sidebar.time_input("End Time", time(20, 0))

# Sensor selection
sensor_list = sorted(locations["Objectnummer"].unique())
selected_sensors = st.sidebar.multiselect("Select Sensors", sensor_list, default=sensor_list[:5])

# Tabs
tab1, tab2 = st.tabs(["Time Series", "Vector Map"])

# ---------------------------------------------------------------
# TAB 1:  TIME SERIES
# ---------------------------------------------------------------

with tab1:
    st.header("Crowd Time Series")

    # Try loading real crowd data (if exists)
    def load_real_data():
        candidates = ["sensordata_SAIL2025.csv", "sensordata.csv", "crowd.csv"]
        for file in candidates:
            try:
                df = pd.read_csv(file)
                return df, file
            except:
                continue
        return None, None

    real_df, real_fname = load_real_data()

    if real_df is not None:
        st.success(f"Using REAL crowd data: {real_fname}")
        real_df.columns = real_df.columns.str.strip()

        # Auto-detect long or wide
        if "sensor" in real_df.columns or "Objectnummer" in real_df.columns:
            # LONG FORMAT
            real_df["timestamp"] = pd.to_datetime(real_df["timestamp"], errors="coerce")
            real_day = real_df[real_df["timestamp"].dt.date == date_sel]
            fig = px.line(real_day, x="timestamp", y="Persons", color="sensor",
                          title="Crowd Levels Over Time (Real Data)")
            st.plotly_chart(fig, use_container_width=True)

        else:
            # WIDE FORMAT → convert to LONG
            tcol = next((c for c in real_df.columns if "time" in c.lower() or "date" in c.lower()), real_df.columns[0])
            real_df["timestamp"] = pd.to_datetime(real_df[tcol], errors="coerce")
            value_cols = [c for c in real_df.columns if c not in ["timestamp", tcol]]

            long_df = real_df.melt(id_vars="timestamp", value_vars=value_cols,
                                   var_name="sensor", value_name="Persons")

            long_df["timestamp"] = pd.to_datetime(long_df["timestamp"])
            real_day = long_df[long_df["timestamp"].dt.date == date_sel]

            fig = px.line(real_day, x="timestamp", y="Persons", color="sensor",
                          title="Crowd Levels Over Time (Real Data)")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Real data not found — showing predicted instead.")
        pred_day = persons_pred[persons_pred["timestamp"].dt.date == date_sel]
        fig = px.line(pred_day, x="timestamp", y="Persons", color="Objectnummer",
                      title="Crowd Levels Over Time (Predicted)")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# TAB 2:  VECTOR MAP
# ---------------------------------------------------------------

with tab2:
    st.header("🗺️ Crowd Movement Vector Map")

    # Filter predicted data by selected date + sensors
    vec_day = vec_long[vec_long["timestamp"].dt.date == date_sel]

    if selected_sensors:
        vec_day = vec_day[vec_day["Objectnummer"].isin(selected_sensors)]

    times_available = sorted(vec_day["timestamp"].unique())

    if not times_available:
        st.warning("No predicted vector data for this date & sensor selection.")
        st.stop()

    # Timestamp selector
    timestamp_sel = st.select_slider(
        "Select Time",
        options=times_available,
        value=times_available[len(times_available) // 2],
        format_func=lambda t: t.strftime("%H:%M")
    )

    scale = st.slider("Arrow Length Scale", 1, 15, 5)

    # Build vector frame
    frame = vec_day[vec_day["timestamp"] == timestamp_sel]
    merged = frame.merge(locations, on="Objectnummer", how="inner")

    if merged.empty:
        st.warning("No matching sensor locations for this selection.")
        st.stop()

    # Compute arrow direction and endpoints
    merged["dx"] = scale * np.sin(np.radians(merged["angle_deg"]))
    merged["dy"] = scale * np.cos(np.radians(merged["angle_deg"]))

    # Plot with Plotly
    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        lat=merged["Lat"],
        lon=merged["Long"],
        mode="markers",
        marker=dict(size=8, color="red"),
        text=merged["Objectnummer"],
        hoverinfo="text"
    ))

    # Add arrows
    for _, row in merged.iterrows():
        fig.add_trace(go.Scattermapbox(
            lat=[row["Lat"], row["Lat"] + row["dy"] * 0.0001],
            lon=[row["Long"], row["Long"] + row["dx"] * 0.0001],
            mode="lines",
            line=dict(width=3, color="orange")
        ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat": merged["Lat"].mean(), "lon": merged["Long"].mean()},
        mapbox_zoom=12,
        margin={"r":0, "t":0, "l":0, "b":0}
    )

    st.plotly_chart(fig, use_container_width=True)


