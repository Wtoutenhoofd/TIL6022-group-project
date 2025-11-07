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
        st.error(f" Required file not found: {path}")
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
tab1, tab2 = st.tabs([" Time Series", " Vector Map"])

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
# TAB 2:  VECTOR MAP (length ∝ value, arrowheads, warnings + P90)
# ---------------------------------------------------------------
with tab2:
    st.header("Crowd Movement Vector Map")

    # --- controls ---
    colA, colB, colC, colD = st.columns(4)
    with colA:
        scale = st.slider("Arrow Length Scale", 1, 20, 5, help="Multiplies arrow length (bigger = longer)")
    with colB:
        show_values = st.toggle("Show values at arrow tips", value=True)
    with colC:
        # keep threshold in session so the 'Suggest' button can set it
        if "warn_threshold" not in st.session_state:
            st.session_state.warn_threshold = 200
        _ = st.empty()  # spacer
        thr = st.number_input("Warn if sensor ≥ people", min_value=1, value=st.session_state.warn_threshold, step=10)
    with colD:
        if st.button("Suggest (P90)"):
            # compute P90 of total people per sensor for selected date
            today = vec_long[vec_long["timestamp"].dt.date == date_sel]
            if today.empty:
                st.info("No data today to suggest a threshold.")
            else:
                p90 = (
                    today.groupby(["timestamp", "Objectnummer"])["value"].sum()
                    .groupby(level=0).sum()  # total across sensors each timestamp
                ).quantile(0.90)
                # If total per timestamp is p90, per-sensor threshold is rougher.
                # Use per-sensor p90 for this date instead:
                per_sensor_p90 = (
                    today.groupby(["timestamp", "Objectnummer"])["value"].sum()
                    .groupby("Objectnummer").quantile(0.90).median()
                )
                # prefer per-sensor p90 (median across sensors) if it yields a sensible number
                suggested = int(max(1, round(per_sensor_p90))) if not np.isnan(per_sensor_p90) else int(max(1, round(p90/5)))
                st.session_state.warn_threshold = suggested
                st.success(f"Suggested threshold set to {suggested}")
        # Read possibly-updated threshold
        warn_threshold = st.session_state.get("warn_threshold", 200)
        # If user typed a new number, keep that as the source of truth
        if thr != warn_threshold:
            warn_threshold = thr
            st.session_state.warn_threshold = thr

    # --- filter data by date and (optionally) sensors ---
    vec_day = vec_long[vec_long["timestamp"].dt.date == date_sel]
    if selected_sensors:
        vec_day = vec_day[vec_day["Objectnummer"].isin(selected_sensors)]

    times_available = sorted(vec_day["timestamp"].unique())
    if not times_available:
        st.warning("No predicted vector data for this date & sensor selection.")
        st.stop()

    # time slider
    timestamp_sel = st.select_slider(
        "Select time",
        options=times_available,
        value=times_available[len(times_available)//2],
        format_func=lambda t: t.strftime("%H:%M")
    )

    # slice time and join coordinates
    frame = vec_day[vec_day["timestamp"] == timestamp_sel].copy()
    merged = frame.merge(locations, on="Objectnummer", how="inner").dropna(subset=["Lat","Long"])

    if merged.empty:
        st.warning("No matching sensor locations for this selection.")
        st.stop()

    # ---- totals per sensor at this time (for warning + labels) ----
    totals = merged.groupby("Objectnummer", as_index=False)["value"].sum().rename(columns={"value":"Persons"})
    over = totals[totals["Persons"] >= warn_threshold]
    if not over.empty:
        sample = ", ".join(over["Objectnummer"].astype(str).tolist()[:8])
        st.error(
            f"⚠️ Crowd alert at {timestamp_sel:%H:%M}: "
            f"{len(over)} sensor(s) ≥ {warn_threshold} people. {('e.g., ' + sample) if sample else ''}"
        )

    # merge totals back for convenient access
    merged = merged.merge(totals, on="Objectnummer", how="left")

    # ---- geometry: arrow length ∝ value, with arrowheads ----
    # 0° = north (up). Use cos for northing (dy), sin for easting (dx).
    # Length units = scale * value; then convert to map "degrees" with DEG_FACTOR.
    merged["dx_units"] = scale * merged["value"] * np.sin(np.radians(merged["angle_deg"]))
    merged["dy_units"] = scale * merged["value"] * np.cos(np.radians(merged["angle_deg"]))

    DEG_FACTOR = 0.0001  # tune alongside 'scale'
    merged["dx_deg"] = merged["dx_units"] * DEG_FACTOR
    merged["dy_deg"] = merged["dy_units"] * DEG_FACTOR

    merged["lat_end"] = merged["Lat"]  + merged["dy_deg"]
    merged["lon_end"] = merged["Long"] + merged["dx_deg"]

    # arrowhead geometry (two short lines from the tip)
    def _rotate(dx, dy, degrees):
        rad = np.radians(degrees)
        return dx*np.cos(rad) - dy*np.sin(rad), dx*np.sin(rad) + dy*np.cos(rad)

    HEAD_ANGLE = 25       # degrees from shaft
    HEAD_FRACTION = 0.35  # head segment length as fraction of shaft length

    hx1, hy1 = _rotate(merged["dx_deg"], merged["dy_deg"], +HEAD_ANGLE)
    hx2, hy2 = _rotate(merged["dx_deg"], merged["dy_deg"], -HEAD_ANGLE)
    merged["hx1_deg"] = hx1 * HEAD_FRACTION
    merged["hy1_deg"] = hy1 * HEAD_FRACTION
    merged["hx2_deg"] = hx2 * HEAD_FRACTION
    merged["hy2_deg"] = hy2 * HEAD_FRACTION

    # ---- draw map ----
    fig = go.Figure()

    # base sensors
    fig.add_trace(go.Scattermapbox(
        lat=merged["Lat"],
        lon=merged["Long"],
        mode="markers",
        marker=dict(size=9, color="#d62728"),  # red
        text=merged["Objectnummer"],
        hovertemplate="<b>%{text}</b><extra></extra>",
        name="Sensor"
    ))

    # shafts + heads (orange)
    for _, r in merged.iterrows():
        # shaft
        fig.add_trace(go.Scattermapbox(
            lat=[r["Lat"], r["lat_end"]],
            lon=[r["Long"], r["lon_end"]],
            mode="lines",
            line=dict(width=max(1, min(7, 1 + 0.4*float(r["value"]))), color="#ff7f0e"),
            hovertemplate=(f"<b>{r['Objectnummer']}</b><br>"
                           f"value: {float(r['value']):.1f}<br>"
                           f"angle: {float(r['angle_deg']):.0f}°<extra></extra>"),
            showlegend=False
        ))
        # head side 1
        fig.add_trace(go.Scattermapbox(
            lat=[r["lat_end"], r["lat_end"] - r["hy1_deg"]],
            lon=[r["lon_end"], r["lon_end"] - r["hx1_deg"]],
            mode="lines",
            line=dict(width=max(1, min(6, 0.8 + 0.3*float(r["value"]))), color="#ff7f0e"),
            hoverinfo="skip",
            showlegend=False
        ))
        # head side 2
        fig.add_trace(go.Scattermapbox(
            lat=[r["lat_end"], r["lat_end"] - r["hy2_deg"]],
            lon=[r["lon_end"], r["lon_end"] - r["hx2_deg"]],
            mode="lines",
            line=dict(width=max(1, min(6, 0.8 + 0.3*float(r["value"]))), color="#ff7f0e"),
            hoverinfo="skip",
            showlegend=False
        ))

    # optional value labels at tip
    if show_values:
        fig.add_trace(go.Scattermapbox(
            lat=merged["lat_end"],
            lon=merged["lon_end"],
            mode="text",
            text=[f"{int(v)}" for v in merged["value"]],
            textfont=dict(size=11, color="#111"),
            hoverinfo="skip",
            showlegend=False
        ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat": float(merged["Lat"].mean()), "lon": float(merged["Long"].mean())},
        mapbox_zoom=12,
        margin={"r":0, "t":6, "l":0, "b":0},
        height=700,
        title=f"Vectors at {timestamp_sel:%Y-%m-%d %H:%M} — length ∝ people"
    )

    st.plotly_chart(fig, use_container_width=True)
