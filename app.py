import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os

# ------------------------------------------------------------------------------------
# Streamlit App: Soil Sense
# ------------------------------------------------------------------------------------
# Comment blocks throughout are in English, explaining each section and function.
# ------------------------------------------------------------------------------------

# Set up the page configuration: title, icon, and layout
st.set_page_config(page_title="Soil Sense", page_icon="🌱", layout="wide")

# ------------------------------------------------------------------------------------
# Custom CSS for a green-themed interface
# ------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main-header {
            color: #2E7D32;
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            color: #2E7D32;
            font-size: 1.2rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        .stSelectbox > div > div {
            border-color: #4CAF50;
        }
        .metric-card {
            background-color: #E8F5E8;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #4CAF50;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------------------------------
# Logo and Title
# ------------------------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Display the Soil Sense logo (make sure attached_assets/soilsense_logo.png exists)
    st.image("attached_assets/soilsense_logo.png", width=300)

st.markdown('<h1 class="main-header">Soil Sense</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">🌱 Agricultural Sensor Data Monitoring Platform</p>',
    unsafe_allow_html=True
)

# ------------------------------------------------------------------------------------
# Section: Data Source Selection
# ------------------------------------------------------------------------------------
st.markdown("---")
st.markdown('<h3 style="color: #2E7D32;">📊 Data Source Selection</h3>', unsafe_allow_html=True)

def get_available_experiments():
    """
    Retrieve a sorted list of available experiment files (Experiment1.txt .. Experiment7.txt)
    from various possible asset paths. Returns names like ["Experiment7", "Experiment6", ...].
    """
    experiments = []
    possible_paths = [
        "./attached_assets",                   # relative path in cloud/Git environment
        "attached_assets",                     # alternative path
        "C:/Users/USER001/Desktop/SoilSense/attached_assets",  # local development path
    ]

    assets_path = None
    for path in possible_paths:
        if os.path.exists(path):
            assets_path = path
            break

    if assets_path:
        # Look for Experiment1 through Experiment7 files (with .txt or .TXT extension)
        for i in range(1, 8):
            filename_txt = f"Experiment{i}.txt"
            filename_TXT = f"Experiment{i}.TXT"
            file_path = None

            if os.path.exists(os.path.join(assets_path, filename_txt)):
                file_path = os.path.join(assets_path, filename_txt)
            elif os.path.exists(os.path.join(assets_path, filename_TXT)):
                file_path = os.path.join(assets_path, filename_TXT)

            if file_path:
                try:
                    with open(file_path, "r") as f:
                        content = f.read().strip()
                        # Ensure the file has some content that resembles timestamp data
                        if content and ('/' in content or ':' in content):
                            experiments.append(f"Experiment{i}")
                except:
                    continue

    # Sort the list in descending order (Experiment7 first, then Experiment6, etc.)
    experiments.sort(key=lambda x: int(x.replace("Experiment", "")), reverse=True)
    return experiments

# Get the list of experiment names that exist
available_experiments = get_available_experiments()

if available_experiments:
    selected_experiment = st.selectbox(
        "Choose data source:", 
        available_experiments, 
        index=0
    )
else:
    st.error(
        "No experiment files found. "
        "Please ensure Experiment1.txt through Experiment7.txt are placed in the attached_assets folder."
    )
    selected_experiment = None

# ------------------------------------------------------------------------------------
# Parsing Function: Reads each line of the TXT and extracts a unified timestamp + sensor values
# ------------------------------------------------------------------------------------
@st.cache_data
def parse_experiment_file(file_path):
    """
    Parse an experiment TXT file where each line begins with "HH:MM:SS DD/MM/YYYY,..." followed by sensor values.
    Steps:
      1. Read the file line by line.
      2. Split the line at the first space: time_part and date+values part.
      3. Construct a combined timestamp "DD/MM/YYYY HH:MM:SS".
      4. Parse that string into a datetime object.
      5. Convert the subsequent comma-separated values into floats.
      6. Return a DataFrame with columns: timestamp, CO2SCD30A [ppm], Temperature_SCD30A [°C], ..., measuredvbat [V].
    """
    try:
        data_rows = []
        with open(file_path, "r") as f:
            lines = f.read().splitlines()

        for line in lines:
            line = line.strip()
            if not line:
                # Skip empty lines
                continue

            # Each line: "16:37:00 07/08/2023,441,28.26,45.69,..."
            if " " not in line:
                # If there's no space, the format is incorrect for our parser
                continue

            # Split into time_part ("16:37:00") and rest ("07/08/2023,441,28.26,...")
            time_part, rest = line.split(" ", 1)
            parts = rest.split(",")

            # First item in parts is date_part ("07/08/2023")
            date_part = parts[0].strip()
            # Construct full timestamp string "07/08/2023 16:37:00"
            timestamp_str = f"{date_part} {time_part}"

            try:
                # Parse with explicit format: day first, then month, then year, and hour:minute:second
                timestamp = datetime.strptime(timestamp_str, "%d/%m/%Y %H:%M:%S")
            except Exception:
                # Skip lines that don't match expected timestamp format
                continue

            # Convert each subsequent part to float (sensor values). If conversion fails, use 0.0
            values = []
            for val in parts[1:]:
                val = val.strip()
                try:
                    values.append(float(val))
                except:
                    values.append(0.0)

            # Append a row: [timestamp, val1, val2, ..., valN]
            data_rows.append([timestamp] + values)

        # Define the column names matching the data order
        columns = [
            "timestamp",
            "CO2SCD30A [ppm]", "Temperature_SCD30A [°C]", "RHSCD30A [%]",
            "CO2SCD30B [ppm]", "Temperature_SCD30B [°C]", "RHSCD30B [%]",
            "CO2SCD30C [ppm]", "Temperature_SCD30C [°C]", "RHSCD30C [%]",
            "CO2SCD30D [ppm]", "Temperature_SCD30D [°C]", "RHSCD30D [%]",
            "oxygenDa_A [%Vol]", "oxygenDa_B [%Vol]", "oxygenDa_C [%Vol]", "oxygenDa_D [%Vol]",
            "oxygenBo_airTemp_A [°C]", "oxygenBo_airTemp_B [°C]", "oxygenBo_airTemp_C [°C]", "oxygenBo_airTemp_D [°C]",
            "measuredvbat [V]"
        ]

        if not data_rows:
            # If no valid rows were parsed, return an empty DataFrame
            return pd.DataFrame()

        # Truncate or match column names to the actual number of columns in data_rows
        max_cols = min(len(columns), len(data_rows[0]))
        df = pd.DataFrame(data_rows, columns=columns[:max_cols])

        # Filter out rows where all sensor values (all columns except timestamp) are zero
        if len(df) > 0:
            df = df[df.iloc[:, 1:].sum(axis=1) > 0]

        # Sort the DataFrame by the timestamp so that plots are chronological
        df.sort_values(by="timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    except Exception as e:
        st.error(f"Error parsing file {file_path}: {e}")
        return pd.DataFrame()

# ------------------------------------------------------------------------------------
# Data Loading Function: Chooses the correct file path based on experiment name
# ------------------------------------------------------------------------------------
@st.cache_data
def load_experiment_data(experiment_name):
    """
    Given an experiment name (e.g., "Experiment3"), try to find "Experiment3.txt" or "Experiment3.TXT"
    in one of the possible asset directories. Then call parse_experiment_file on the found path.
    """
    if not experiment_name:
        return pd.DataFrame()

    possible_paths = [
        "attached_assets",                                    # cloud environment
        "C:/Users/USER001/Desktop/SoilSense/attached_assets",  # local development path
        "./attached_assets"                                   # relative path
    ]

    for base_path in possible_paths:
        if os.path.exists(base_path):
            txt_path = os.path.join(base_path, f"{experiment_name}.txt")
            TXT_path = os.path.join(base_path, f"{experiment_name}.TXT")
            if os.path.exists(txt_path):
                return parse_experiment_file(txt_path)
            elif os.path.exists(TXT_path):
                return parse_experiment_file(TXT_path)

    # If no file was found in any of the paths, return an empty DataFrame
    return pd.DataFrame()

# ------------------------------------------------------------------------------------
# Function to create separate Plotly line charts for each sensor category
# ------------------------------------------------------------------------------------
def create_sensor_plots(df):
    """
    Given a DataFrame df with timestamp and various sensor columns, build a list of
    (title, figure) tuples for CO2, Temperature, Humidity, Oxygen, and Oxygen Temperature.
    Returns an empty list if df is empty.
    """
    if df.empty:
        return []

    # Identify sensor columns by prefix
    co2_sensors = [col for col in df.columns if col.startswith("CO2")]
    temp_sensors = [col for col in df.columns if col.startswith("Temperature")]
    humidity_sensors = [col for col in df.columns if col.startswith("RH")]
    oxygen_sensors = [col for col in df.columns if col.startswith("oxygenDa")]
    oxygen_temp_sensors = [col for col in df.columns if col.startswith("oxygenBo_airTemp")]

    plots = []

    # CO2 Plot
    if co2_sensors:
        fig_co2 = go.Figure()
        for sensor in co2_sensors:
            fig_co2.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df[sensor],
                    mode="lines",
                    name=sensor,
                    line=dict(width=2)
                )
            )
        fig_co2.update_layout(
            title="🌱 CO2 Levels (ppm)",
            xaxis_title="Time",
            yaxis_title="CO2 (ppm)",
            hovermode="x unified",
            height=400
        )
        plots.append(("CO2 Sensors", fig_co2))

    # Temperature Plot
    if temp_sensors:
        fig_temp = go.Figure()
        for sensor in temp_sensors:
            fig_temp.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df[sensor],
                    mode="lines",
                    name=sensor,
                    line=dict(width=2)
                )
            )
        fig_temp.update_layout(
            title="🌡️ Temperature Readings (°C)",
            xaxis_title="Time",
            yaxis_title="Temperature (°C)",
            hovermode="x unified",
            height=400
        )
        plots.append(("Temperature Sensors", fig_temp))

    # Humidity Plot
    if humidity_sensors:
        fig_humidity = go.Figure()
        for sensor in humidity_sensors:
            fig_humidity.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df[sensor],
                    mode="lines",
                    name=sensor,
                    line=dict(width=2)
                )
            )
        fig_humidity.update_layout(
            title="💧 Humidity Levels (%)",
            xaxis_title="Time",
            yaxis_title="Humidity (%)",
            hovermode="x unified",
            height=400
        )
        plots.append(("Humidity Sensors", fig_humidity))

    # Oxygen Plot
    if oxygen_sensors:
        fig_oxygen = go.Figure()
        for sensor in oxygen_sensors:
            fig_oxygen.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df[sensor],
                    mode="lines",
                    name=sensor,
                    line=dict(width=2)
                )
            )
        fig_oxygen.update_layout(
            title="🫁 Oxygen Levels (%Vol)",
            xaxis_title="Time",
            yaxis_title="Oxygen (%Vol)",
            hovermode="x unified",
            height=400
        )
        plots.append(("Oxygen Sensors", fig_oxygen))

    # Oxygen Temperature Plot
    if oxygen_temp_sensors:
        fig_oxygen_temp = go.Figure()
        for sensor in oxygen_temp_sensors:
            fig_oxygen_temp.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df[sensor],
                    mode="lines",
                    name=sensor,
                    line=dict(width=2)
                )
            )
        fig_oxygen_temp.update_layout(
            title="🌡️ Oxygen Sensor Temperature (°C)",
            xaxis_title="Time",
            yaxis_title="Temperature (°C)",
            hovermode="x unified",
            height=400
        )
        plots.append(("Oxygen Temperature", fig_oxygen_temp))

    return plots

# ------------------------------------------------------------------------------------
# Function to create a correlation heatmap using Plotly Express
# ------------------------------------------------------------------------------------
def create_correlation_heatmap(df, columns):
    """
    Given a DataFrame df and a list of column names, compute the Pearson correlation matrix
    and return a heatmap figure. Returns None if fewer than 2 columns are selected.
    """
    if len(columns) < 2:
        return None

    corr_df = df[columns].corr()
    fig = px.imshow(
        corr_df,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Sensor Correlation Heatmap"
    )
    fig.update_layout(height=400)
    return fig

# ------------------------------------------------------------------------------------
# Function to check sensor thresholds and generate alert messages
# ------------------------------------------------------------------------------------
def check_sensor_thresholds(df, thresholds):
    """
    Given the latest row of df and a thresholds dictionary, compare each sensor reading
    to its min/max thresholds and return a list of alert dictionaries.
    Each alert dict contains: sensor nam
