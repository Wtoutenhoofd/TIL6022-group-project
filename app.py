# app.py
import re
import pandas as pd
import streamlit as st
import plotly.express as px

import re
import json
import numpy as np
import plotly.graph_objects as go
from urllib.request import urlopen

# ---------------- CONFIG (filenames) ----------------
LOC_FILE   = "sensor-location.xlsx - Sheet1.csv"   # has Objectnummer + Lat/Long
CROWD_FILE = "sensordata_SAIL2025.csv"             # wide: each sensor = a column
PRED_FILE   = "predicted_sensor_values_3min.csv"   # same file as in your screenshots
LOC_FILE    = "sensor-location.xlsx - Sheet1.csv"  # already in your app
GEOJSON_URL = "https://maps.amsterdam.nl/open_geodata/geojson_lnglat.php?KAARTLAAG=INDELING_STADSDEEL&THEMA=gebiedsindeling"

# ---------------------------------------------------

st.set_page_config(page_title="SAIL Crowd Monitor", layout="wide")

# ---------- helpers ----------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

@st.cache_data
def load_geojson(url=GEOJSON_URL):
    with urlopen(url) as resp:
        return json.load(resp)

@st.cache_data
def load_locations_vector(path=LOC_FILE) -> pd.DataFrame:
    loc = pd.read_csv(path, quotechar='"')
    loc.columns = [c.strip() for c in loc.columns]
    n2o = {_norm(c): c for c in loc.columns}
    # sensor id
    sensor_col = None
    for k in ("objectnummer","sensor","sensorid","id","objectnr"):
        if k in n2o: sensor_col = n2o[k]; break
    if sensor_col is None:
        raise ValueError(f"Could not find sensor id column in {path}. Columns: {list(loc.columns)}")
    # lat/long (combined or separate)
    ll = n2o.get("latlong") or n2o.get("latlon")
    if ll:
        parts = loc[ll].astype(str).str.split(",", n=1, expand=True)
        loc["Lat"]  = pd.to_numeric(parts[0].str.replace('"','').str.strip(), errors="coerce")
        loc["Long"] = pd.to_numeric(parts[1].str.replace('"','').str.strip(), errors="coerce")
    else:
        lat_col = next((n2o[k] for k in n2o if k.startswith("lat") or "latitude" in k), None)
        lon_col = next((n2o[k] for k in n2o if k.startswith("lon") or "long" in k or "longitude" in k), None)
        loc["Lat"]  = pd.to_numeric(loc[lat_col], errors="coerce")
        loc["Long"] = pd.to_numeric(loc[lon_col], errors="coerce")

    loc = loc.rename(columns={sensor_col: "Objectnummer"})
    return loc[["Objectnummer","Lat","Long"]].dropna(subset=["Lat","Long"])

def _parse_sensor_angle(col: str):
    """Split 'CMSA-GAWW-11_120' -> ('CMSA-GAWW-11', 120). Returns (name, None) if no angle."""
    if col == "timestamp": return (col, None)
    m = re.match(r"^(.*)_(\d+)$", col)
    if not m: 
        return (col, None)
    return (m.group(1), int(m.group(2)))

@st.cache_data
def load_predicted_vectors(path=PRED_FILE) -> pd.DataFrame:
    """
    Load predicted sensor values (3-min file in your screenshots).
    - merges '-B_ANGLE' columns into main columns (sum),
    - returns long df with [timestamp, Objectnummer, hoek, value].
    """
    cs = pd.read_csv(path, parse_dates=["timestamp"])
    cs.columns = [c.strip() for c in cs.columns]

    # 1) merge B-suffix columns: e.g. 'GASA-06_95' + 'GASA-06-B_95'
    cols = cs.columns.tolist()
    for col in cols:
        if col.endswith("_B") or col.endswith("-B"):  # guard for weird names
            # not our pattern; we only care about '-B_angle'
            continue
    # Find all B columns by pattern '(.*)-B_(angle)'
    b_cols = [c for c in cs.columns if re.match(r"^(.+)-B_(\d+)$", c)]
    for bc in b_cols:
        m = re.match(r"^(.+)-B_(\d+)$", bc)
        base = m.group(1)
        ang  = m.group(2)
        main = f"{base}_{ang}"
        if main in cs.columns:
            cs[main] = cs[main].fillna(0) + cs[bc].fillna(0)
            cs.drop(columns=[bc], inplace=True)

    # 2) melt to long
    sensor_cols = [c for c in cs.columns if c != "timestamp"]
    long = cs.melt(id_vars="timestamp", value_vars=sensor_cols,
                   var_name="column", value_name="value")
    # parse object + angle
    parsed = long["column"].apply(_parse_sensor_angle)
    long["Objectnummer"] = parsed.apply(lambda x: x[0])
    long["hoek"]         = parsed.apply(lambda x: x[1])
    # keep only rows with angle present
    long = long[long["hoek"].notna()].copy()
    long["hoek"]  = long["hoek"].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce").fillna(0.0)
    return long[["timestamp","Objectnummer","hoek","value"]].sort_values(["timestamp","Objectnummer","hoek"])

def build_vector_frame(long_df: pd.DataFrame, loc_df: pd.DataFrame, t0: pd.Timestamp,
                       scale_to_m: float = 3.0, zero_is_north: bool = True) -> pd.DataFrame:
    """
    For a given timestamp t0, compute arrow endpoints for each (sensor, hoek).
    - scale_to_m: visual scale: meters per unit 'value'
    - zero_is_north: if True, interpret angles as compass (0°=north, CW positive)
    Returns df with Lat/Long and Lat_end/Long_end and color by towards/away if target is set later.
    """
    use = long_df[long_df["timestamp"] == t0].merge(loc_df, on="Objectnummer", how="left").dropna(subset=["Lat","Long"])
    if use.empty:
        return use

    # meters per degree at each sensor (small offset approx)
    lat_rad = np.deg2rad(use["Lat"].values)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(lat_rad)

    # direction in radians
    raw_deg = use["hoek"].astype(float).values
    theta   = np.deg2rad(90.0 - raw_deg) if zero_is_north else np.deg2rad(raw_deg)

    length_m = use["value"].astype(float).values * scale_to_m
    dx_m = length_m * np.cos(theta)   # east(+)/west(-)
    dy_m = length_m * np.sin(theta)   # north(+)/south(-)

    dlat = dy_m / m_per_deg_lat
    dlon = dx_m / m_per_deg_lon

    use = use.copy()
    use["Lat_end"]  = use["Lat"].values  + dlat
    use["Long_end"] = use["Long"].values + dlon
    return use


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

@st.cache_data
def load_locations(path: str) -> pd.DataFrame:
    # read with potential quotes inside lat/long
    loc = pd.read_csv(path, quotechar='"')
    loc.columns = [c.strip() for c in loc.columns]
    n2o = {_norm(c): c for c in loc.columns}

    # detect sensor id column
    sensor_col = None
    for key in ("objectnummer","objectnr","sensor","sensorid","id"):
        if key in n2o:
            sensor_col = n2o[key]; break
    if sensor_col is None:
        st.error(f"Could not find sensor ID column in {path}. Columns: {list(loc.columns)}")
        st.stop()

    # combined or separate lat/long
    latlong_col = n2o.get("latlong") or n2o.get("latlon")
    if latlong_col:
        parts = loc[latlong_col].astype(str).str.split(",", n=1, expand=True)
        loc["Lat"]  = pd.to_numeric(parts[0].str.replace('"','').str.strip(), errors="coerce")
        loc["Long"] = pd.to_numeric(parts[1].str.replace('"','').str.strip(), errors="coerce")
    else:
        lat_col = next((n2o[k] for k in n2o if k.startswith("lat") or "latitude" in k), None)
        lon_col = next((n2o[k] for k in n2o if k.startswith("lon") or "long" in k or "longitude" in k), None)
        loc["Lat"]  = pd.to_numeric(loc[lat_col], errors="coerce")
        loc["Long"] = pd.to_numeric(loc[lon_col], errors="coerce")

    loc = loc.rename(columns={sensor_col: "sensor"})
    return loc[["sensor","Lat","Long"]].dropna(subset=["Lat","Long"])

@st.cache_data
def load_crowd_wide_to_long(path: str) -> pd.DataFrame:
    cs = pd.read_csv(path)
    cs.columns = [c.strip() for c in cs.columns]
    # detect timestamp column or create one (assume 10-min cadence from 2025-08-20)
    tcol = next((c for c in cs.columns if _norm(c) in
                {"timestamp","time","datetime","date","datetimestamp"}), None)
    if tcol is None:
        cs.insert(0, "timestamp", pd.date_range("2025-08-20 00:00", periods=len(cs), freq="10min"))
    else:
        cs["timestamp"] = pd.to_datetime(cs[tcol], errors="coerce", infer_datetime_format=True)

    sensor_cols = [c for c in cs.columns if c != "timestamp"]
    long_df = cs.melt(id_vars="timestamp", value_vars=sensor_cols,
                      var_name="sensor", value_name="Persons")
    long_df["Persons"] = pd.to_numeric(long_df["Persons"], errors="coerce").fillna(0.0)
    return long_df.dropna(subset=["timestamp"]).sort_values(["timestamp","sensor"])

def resample_people(df: pd.DataFrame, rule: str, how="sum"):
    # df: columns = timestamp, sensor, Persons
    if rule == "raw":
        return df.copy()
    if how == "sum":
        agg = df.set_index("timestamp").groupby("sensor")["Persons"].resample(rule).sum().reset_index()
    else:
        agg = df.set_index("timestamp").groupby("sensor")["Persons"].resample(rule).mean().reset_index()
    return agg

# ---------- load ----------
loc = load_locations(LOC_FILE)
crowd = load_crowd_wide_to_long(CROWD_FILE)

# enrich with lat/long for convenience
crowd = crowd.merge(loc, on="sensor", how="left")

# ---------- sidebar filters ----------
st.sidebar.header("Filters")
all_dates = sorted(crowd["timestamp"].dt.date.unique())
date_sel = st.sidebar.selectbox("Date", all_dates, index=0)

# time window
day_df = crowd[crowd["timestamp"].dt.date == date_sel]
min_t = day_df["timestamp"].min()
max_t = day_df["timestamp"].max()
start_t = st.sidebar.time_input("Start", pd.to_datetime("09:00").time())
end_t   = st.sidebar.time_input("End"  , pd.to_datetime("20:00").time())

window = day_df[(day_df["timestamp"].dt.time >= start_t) &
                (day_df["timestamp"].dt.time <= end_t)]

# sensors
sensors_available = sorted(window["sensor"].dropna().unique().tolist())
selected_sensors = st.sidebar.multiselect("Sensors", sensors_available, default=sensors_available[:6])

# aggregation
res_rule = st.sidebar.selectbox("Aggregate to", ["raw","5min","10min","15min","30min","60min"], index=2)
agg_method = st.sidebar.selectbox("Aggregate method", ["sum","mean"], index=0)

# view options
stack_mode = st.sidebar.selectbox("Line chart mode", ["overlay","stacked"], index=0)
show_total = st.sidebar.checkbox("Show overall total line", value=True)

# apply filters
window = window[window["sensor"].isin(selected_sensors)]
agg = resample_people(window, res_rule, how=agg_method)

# ---------- charts ----------
st.title("SAIL Crowd Monitor — People over Time")

# === VECTOR MAP TAB (like your teammate) ===
st.divider()
st.header("🗺️ Vector Map (per sensor / angle)")

# load once
stadsdelen = load_geojson()
loc_vec    = load_locations_vector()
pred_long  = load_predicted_vectors()

# Choose time: use currently filtered day window if you have it; otherwise, the last in file
# If your app already has 'date_sel', 'start_t', 'end_t', reuse them; else show a picker here.
pred_day = pred_long[pred_long["timestamp"].dt.date == date_sel] if 'date_sel' in locals() else pred_long
if 'start_t' in locals() and 'end_t' in locals():
    pred_day = pred_day[(pred_day["timestamp"].dt.time >= start_t) & (pred_day["timestamp"].dt.time <= end_t)]

available_times = sorted(pred_day["timestamp"].unique())
if not available_times:
    st.info("No vector data for this date/time selection.")
else:
    t0_idx = len(available_times) - 1  # latest
    t0 = st.select_slider("Pick timestamp", options=available_times, value=available_times[t0_idx],
                          format_func=lambda x: x.strftime("%Y-%m-%d %H:%M"))

    colA, colB, colC = st.columns(3)
    with colA:
        scale_to_m = st.slider("Arrow length scale (m per unit)", 1.0, 10.0, 3.0, 0.5)
    with colB:
        zero_is_north = st.toggle("Angles are compass (0°=north, CW+)", value=True)
    with colC:
        show_legend = st.toggle("Show length legend", value=True)

    vec_df = build_vector_frame(pred_long, loc_vec, t0, scale_to_m=scale_to_m, zero_is_north=zero_is_north)

    if vec_df.empty:
        st.warning("No rows for that timestamp.")
    else:
        # --- Plotly figure (same style as screenshots: Scattermapbox + GEOJSON line layer) ---
        fig = go.Figure()

        # (a) sensor markers
        bases = vec_df[["Objectnummer","Lat","Long"]].drop_duplicates()
        fig.add_trace(go.Scattermapbox(
            lat=bases["Lat"], lon=bases["Long"],
            mode="markers",
            marker=dict(size=8, color="#1f77b4"),
            text=bases["Objectnummer"],
            hovertemplate="<b>%{text}</b><extra></extra>",
            name="Sensor"
        ))

        # (b) vector lines (one trace per row to keep hover clean)
        # line width scaled by value, minimum 1
        for _, r in vec_df.iterrows():
            fig.add_trace(go.Scattermapbox(
                lat=[r["Lat"], r["Lat_end"]],
                lon=[r["Long"], r["Long_end"]],
                mode="lines",
                line=dict(width=max(1, min(6, 1 + 0.5*float(r["value"]))), color="#222"),
                hovertemplate=(f"<b>{r['Objectnummer']}</b><br>"
                               f"hoek: {int(r['hoek'])}°<br>"
                               f"value: {float(r['value']):.2f}<extra></extra>"),
                showlegend=False
            ))

        # (c) optional length legend (fixed angle eastward), bottom-left of view
        if show_legend:
            legend_vals = [10, 25, 50]  # tweak as you like
            # anchor near the overall center
            anchor_lat = float(np.nanmean(vec_df["Lat"]))
            anchor_lon = float(np.nanmean(vec_df["Long"])) - 0.03
            # small offsets
            m_per_deg_lat = 111_320.0
            m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(anchor_lat))
            legend_theta = 0.0 if not zero_is_north else np.deg2rad(90.0 - 90.0)  # pointing east
            for i, v in enumerate(legend_vals):
                Lm = v * scale_to_m
                dx = Lm * np.cos(legend_theta) / m_per_deg_lon
                dy = Lm * np.sin(legend_theta) / m_per_deg_lat
                lat0 = anchor_lat + i * 0.0015
                lon0 = anchor_lon
                fig.add_trace(go.Scattermapbox(
                    lat=[lat0, lat0 + dy], lon=[lon0, lon0 + dx],
                    mode="lines", line=dict(color="black", width=3),
                    name=f"{v} persons",
                    showlegend=True, hoverinfo="skip"
                ))

        # layout + Amsterdam districts overlay
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox=dict(
                center=dict(lat=52.372, lon=4.900),
                zoom=10,
                layers=[dict(source=load_geojson(), below='', sourcetype="geojson",
                             type="line", color="black", line=dict(width=1))]
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            title=f"Pijlen per sensor/hoek — {t0:%Y-%m-%d %H:%M}"
        )

        st.plotly_chart(fig, use_container_width=True)


# KPI row
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Date", str(date_sel))
with c2: st.metric("Sensors selected", len(selected_sensors))
with c3: st.metric("Time range", f"{start_t.strftime('%H:%M')}–{end_t.strftime('%H:%M')}")
with c4: st.metric("Frames (points)", agg["timestamp"].nunique())

# line chart (per sensor)
fig = px.line(
    agg, x="timestamp", y="Persons", color="sensor",
    labels={"timestamp":"Time","Persons":"People"}
)
if stack_mode == "stacked":
    # switch to area chart stacked
    fig = px.area(
        agg, x="timestamp", y="Persons", color="sensor",
        groupnorm=None
    )
fig.update_layout(legend_title_text="Sensor", hovermode="x unified", height=450)

# add a total line if requested
if show_total:
    tot = agg.groupby("timestamp", as_index=False)["Persons"].sum()
    fig.add_scatter(x=tot["timestamp"], y=tot["Persons"], name="Total", mode="lines", line=dict(width=3))

st.plotly_chart(fig, use_container_width=True)

# Top locations in the window
top = (agg.groupby("sensor", as_index=False)["Persons"].sum()
          .sort_values("Persons", ascending=False))
top = top.merge(loc, left_on="sensor", right_on="sensor", how="left")
st.subheader("Top locations in selection")
st.dataframe(top.rename(columns={"Persons":"People (agg)"}), use_container_width=True)

# Map of selected sensors with bubble size = average people in range
avg = (
    agg.groupby("sensor", as_index=False)["Persons"].mean()   # no Lat/Long here yet
      .merge(loc, on="sensor", how="left")                    # add coordinates
      .dropna(subset=["Lat","Long"])
)
avg["Persons"] = avg["Persons"].clip(lower=0).round(1)

map_fig = px.scatter_mapbox(
    avg,
    lat="Lat", lon="Long",
    size="Persons", color="sensor",
    hover_name="sensor",
    zoom=12, height=500,
    mapbox_style="open-street-map"
)
st.subheader("Sensors map (bubble size = avg people in selected window)")
st.plotly_chart(map_fig, use_container_width=True)


# optional export
st.download_button("Download filtered time series (CSV)",
                   data=agg.to_csv(index=False).encode("utf-8"),
                   file_name=f"sail_crowd_timeseries_{date_sel}.csv",
                   mime="text/csv")


