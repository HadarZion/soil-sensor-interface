# SOIL SENSE DEPLOYMENT PACKAGE

## File 1: main_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Soil Sense",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS for green theme
st.markdown("""
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
""", unsafe_allow_html=True)

# Logo and Title
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("attached_assets/soilsense_logo.png", width=300)

st.markdown('<h1 class="main-header">Soil Sense</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">🌱 Agricultural Sensor Data Monitoring Platform</p>', unsafe_allow_html=True)

# Data source section
st.markdown("---")
st.markdown('<h3 style="color: #2E7D32;">☁ Data Source Options</h3>', unsafe_allow_html=True)

# Options for data source
data_source = st.radio(
    "Choose data source:",
    ["📊 Use sample data", "📁 Upload file manually", "🔄 Connect to cloud storage"],
    index=0
)

uploaded_file = None
if data_source == "📁 Upload file manually":
    uploaded_file = st.file_uploader(
        "Upload a TXT file from your sensor data",
        type=['txt'],
        help="Select a sensor data file from your agricultural monitoring system"
    )


# Function to parse uploaded TXT file
@st.cache_data
def parse_uploaded_file(file_content):
    """Parse uploaded TXT file content"""
    try:
        # Convert bytes to string if needed
        if isinstance(file_content, bytes):
            content = file_content.decode('utf-8')
        else:
            content = file_content

        lines = content.strip().split('\n')

        # Skip the header line and parse data
        data_rows = []
        for line in lines[1:]:  # Skip header
            parts = line.strip().split(',')
            if len(parts) >= 11:  # Ensure we have enough columns
                try:
                    # Parse timestamp from "HH:MM:SS DD/MM/YYYY" format
                    time_date = parts[0].strip()
                    if '/' in time_date and ':' in time_date:
                        # Handle different date formats
                        if '148/03/2000' in time_date or 'invalid' in time_date.lower():
                            continue  # Skip corrupted entries

                        timestamp = datetime.strptime(time_date, '%H:%M:%S %d/%m/%Y')

                        # Parse sensor values
                        values = [float(x.strip()) for x in parts[1:]]

                        data_rows.append([timestamp] + values)
                except:
                    continue  # Skip problematic lines

        # Create DataFrame with proper sensor names
        columns = ['timestamp', 'CO2SCD30A_ppm', 'TemperatureSCD30A_C', 'RHSCD30A_%',
                   'CO2SCD30B_ppm', 'TemperatureSCD30B_C', 'RHSCD30B_%', 'CO2SCD30C_ppm',
                   'TemperatureSCD30C_C', 'RHSCD30C_%', 'Battery_Volt']

        df = pd.DataFrame(data_rows, columns=columns)

        # Filter out inactive periods
        df = df[(df['CO2SCD30A_ppm'] > 0) | (df['TemperatureSCD30A_C'] > 0) | (df['Battery_Volt'] > 4.0)]

        return df

    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return pd.DataFrame()


# Function to load sample data
@st.cache_data
def load_sample_data():
    """Load sample data from attached file"""
    try:
        with open('attached_assets/SOILCO2.TXT', 'r') as file:
            content = file.read()
        return parse_uploaded_file(content)
    except:
        return pd.DataFrame()


# Cache the data generation to improve performance
@st.cache_data
def load_data_source():
    """Load data based on selected source"""
    if data_source == "📊 Use sample data":
        return load_sample_data()
    elif data_source == "📁 Upload file manually" and uploaded_file is not None:
        file_content = uploaded_file.read()
        return parse_uploaded_file(file_content)
    elif data_source == "🔄 Connect to cloud storage":
        st.info("🔄 Cloud storage connection ready for configuration")
        return load_sample_data()
    else:
        return pd.DataFrame()


# Function to create separate plots by sensor type
def create_sensor_plots(df):
    """Create separate plots for CO2, Temperature, and Humidity sensors"""
    if df.empty:
        return []

    # Group sensors by type
    co2_sensors = [col for col in df.columns if col.startswith('CO2')]
    temp_sensors = [col for col in df.columns if col.startswith('Temperature')]
    humidity_sensors = [col for col in df.columns if col.startswith('RH')]

    plots = []

    # CO2 Plot
    if co2_sensors:
        fig_co2 = go.Figure()
        for sensor in co2_sensors:
            fig_co2.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df[sensor],
                mode='lines',
                name=sensor.replace('_', ' '),
                line=dict(width=2)
            ))
        fig_co2.update_layout(
            title="🌱 CO2 Levels (ppm)",
            xaxis_title="Time",
            yaxis_title="CO2 (ppm)",
            hovermode='x unified',
            height=400
        )
        plots.append(("CO2 Sensors", fig_co2))

    # Temperature Plot
    if temp_sensors:
        fig_temp = go.Figure()
        for sensor in temp_sensors:
            fig_temp.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df[sensor],
                mode='lines',
                name=sensor.replace('_', ' '),
                line=dict(width=2)
            ))
        fig_temp.update_layout(
            title="🌡 Temperature Readings (°C)",
            xaxis_title="Time",
            yaxis_title="Temperature (°C)",
            hovermode='x unified',
            height=400
        )
        plots.append(("Temperature Sensors", fig_temp))

    # Humidity Plot
    if humidity_sensors:
        fig_humidity = go.Figure()
        for sensor in humidity_sensors:
            fig_humidity.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df[sensor],
                mode='lines',
                name=sensor.replace('_', ' '),
                line=dict(width=2)
            ))
        fig_humidity.update_layout(
            title="💧 Humidity Levels (%)",
            xaxis_title="Time",
            yaxis_title="Humidity (%)",
            hovermode='x unified',
            height=400
        )
        plots.append(("Humidity Sensors", fig_humidity))

    return plots


# Create correlation heatmap
def create_correlation_heatmap(df, columns):
    """Create a correlation heatmap for selected sensors"""
    if len(columns) < 2:
        return None

    corr_df = df[columns].corr()

    fig = px.imshow(
        corr_df,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title="Sensor Correlation Heatmap"
    )

    fig.update_layout(height=400)
    return fig


# Alert System Functions
def check_sensor_thresholds(df, thresholds):
    """Check current sensor readings against defined thresholds"""
    if df.empty or len(df) == 0:
        return []

    alerts = []
    latest_row = df.iloc[-1]
    current_time = datetime.now()

    # Check CO2 sensors
    co2_columns = [col for col in df.columns if col.startswith('CO2')]
    for col in co2_columns:
        value = latest_row[col]
        if value < thresholds['co2']['min']:
            alerts.append({
                'sensor': col,
                'message': f'CO2 level too low',
                'current_value': value,
                'threshold': thresholds['co2']['min'],
                'severity': 'medium',
                'timestamp': current_time
            })
        elif value > thresholds['co2']['max']:
            alerts.append({
                'sensor': col,
                'message': f'CO2 level too high',
                'current_value': value,
                'threshold': thresholds['co2']['max'],
                'severity': 'high',
                'timestamp': current_time
            })

    # Check Temperature sensors
    temp_columns = [col for col in df.columns if col.startswith('Temperature')]
    for col in temp_columns:
        value = latest_row[col]
        if value < thresholds['temperature']['min']:
            alerts.append({
                'sensor': col,
                'message': f'Temperature too low',
                'current_value': value,
                'threshold': thresholds['temperature']['min'],
                'severity': 'high',
                'timestamp': current_time
            })
        elif value > thresholds['temperature']['max']:
            alerts.append({
                'sensor': col,
                'message': f'Temperature too high',
                'current_value': value,
                'threshold': thresholds['temperature']['max'],
                'severity': 'high',
                'timestamp': current_time
            })

    # Check Humidity sensors
    humidity_columns = [col for col in df.columns if col.startswith('RH')]
    for col in humidity_columns:
        value = latest_row[col]
        if value < thresholds['humidity']['min']:
            alerts.append({
                'sensor': col,
                'message': f'Humidity too low',
                'current_value': value,
                'threshold': thresholds['humidity']['min'],
                'severity': 'medium',
                'timestamp': current_time
            })
        elif value > thresholds['humidity']['max']:
            alerts.append({
                'sensor': col,
                'message': f'Humidity too high',
                'current_value': value,
                'threshold': thresholds['humidity']['max'],
                'severity': 'medium',
                'timestamp': current_time
            })

    # Check Battery voltage
    if 'Battery_Volt' in df.columns:
        value = latest_row['Battery_Volt']
        if value < thresholds['battery']['min']:
            alerts.append({
                'sensor': 'Battery_Volt',
                'message': f'Battery voltage low',
                'current_value': value,
                'threshold': thresholds['battery']['min'],
                'severity': 'high',
                'timestamp': current_time
            })

    return alerts


# Load data based on user selection
df = load_data_source()

if not df.empty:
    # Display data source info
    if data_source == "📊 Use sample data":
        st.info("📊 Using sample agricultural sensor data")
    elif data_source == "📁 Upload file manually" and uploaded_file is not None:
        st.success(f"✅ Data loaded from: {uploaded_file.name}")
    else:
        st.info("🔄 Cloud storage connection ready")

    # Main application interface
    st.markdown("---")
    st.markdown('<h3 style="color: #2E7D32;">📈 Soil Sense Data Analysis</h3>', unsafe_allow_html=True)

    # Display data info
    st.write(f"📊 *Records*: {len(df)}")
    if 'timestamp' in df.columns and len(df) > 0:
        st.write(
            f"⏰ *Time Range*: {df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")

    # Get sensor columns (all except timestamp)
    sensor_columns = [col for col in df.columns if col != 'timestamp']

    # Create 4 tabs for different visualizations
    viz_tab1, viz_tab2, viz_tab3, alert_tab = st.tabs(
        ["📈 Time Series", "📊 Statistics", "🔄 Correlations", "🚨 Alert System"])

    with viz_tab1:
        # Time range selection
        st.subheader("Time Range Selection")

        # Get min and max timestamps
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()

        # Create two columns for start and end time
        col1, col2 = st.columns(2)

        with col1:
            start_time = st.date_input(
                "Start Date",
                value=min_time.date(),
                min_value=min_time.date(),
                max_value=max_time.date()
            )
            start_hour = st.time_input(
                "Start Time",
                value=min_time.time()
            )

        with col2:
            end_time = st.date_input(
                "End Date",
                value=max_time.date(),
                min_value=min_time.date(),
                max_value=max_time.date()
            )
            end_hour = st.time_input(
                "End Time",
                value=max_time.time()
            )

        # Combine date and time inputs
        start_datetime = pd.to_datetime(f"{start_time} {start_hour}")
        end_datetime = pd.to_datetime(f"{end_time} {end_hour}")

        # Filter dataframe based on selected time range
        if start_datetime <= end_datetime:
            filtered_df = df[
                (df['timestamp'] >= start_datetime) &
                (df['timestamp'] <= end_datetime)
                ]

            if not filtered_df.empty:
                # Create separate plots by sensor type
                plots = create_sensor_plots(filtered_df)

                # Display each plot type
                for plot_title, fig in plots:
                    st.subheader(plot_title)
                    st.plotly_chart(fig, use_container_width=True)

                # Show data summary for filtered range
                st.write(f"📊 Showing {len(filtered_df)} data points from {start_time} to {end_time}")
            else:
                st.warning("No data found in the selected time range")
        else:
            st.error("Start time must be before end time")

    with viz_tab2:
        st.subheader("Sensor Statistics")

        # Create a dataframe with statistics for all sensors
        stats = []
        for col in sensor_columns:
            stats.append({
                "Sensor": col,
                "Min": f"{df[col].min():.1f}",
                "Max": f"{df[col].max():.1f}",
                "Average": f"{df[col].mean():.1f}",
                "Std Dev": f"{df[col].std():.1f}"
            })

        # Display statistics as a table
        st.table(pd.DataFrame(stats))

        # Show the latest readings
        st.subheader("Latest Readings")
        latest = df.iloc[-1:].copy()
        latest_time = latest['timestamp'].iloc[0]
        st.write(f"Time: {latest_time}")

        # Create metrics in columns
        cols = st.columns(len(sensor_columns))
        for i, sensor in enumerate(sensor_columns):
            if len(df) >= 2:
                prev_value = df.iloc[-2][sensor]
                current_value = latest[sensor].iloc[0]
                delta = current_value - prev_value
            else:
                current_value = latest[sensor].iloc[0]
                delta = 0

            with cols[i]:
                st.metric(
                    label=sensor,
                    value=f"{current_value:.1f}",
                    delta=f"{delta:.1f}"
                )

    with viz_tab3:
        st.subheader("Sensor Correlations")

        # Select sensors for correlation
        correlation_sensors = st.multiselect(
            "Select sensors for correlation analysis:",
            sensor_columns,
            default=sensor_columns if len(sensor_columns) <= 4 else sensor_columns[:4]
        )

        if len(correlation_sensors) >= 2:
            # Create correlation heatmap
            corr_fig = create_correlation_heatmap(df, correlation_sensors)
            st.plotly_chart(corr_fig, use_container_width=True)

            # Explanation
            st.info("""
            Correlation values range from -1 to 1:
            * Values close to 1 indicate a strong positive correlation
            * Values close to -1 indicate a strong negative correlation  
            * Values close to 0 indicate little to no correlation
            """)
        else:
            st.info("Please select at least two sensors to view correlations")

    # Option to view raw data
    with st.expander("View Raw Data", expanded=False):
        st.subheader("Sample of Raw Data")
        st.dataframe(df.head(20), use_container_width=True)

        # Option to download data
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Full Dataset",
            data=csv,
            file_name="soil_sense_data.csv",
            mime="text/csv"
        )

    with alert_tab:
        st.markdown('<h3 style="color: #2E7D32;">🚨 Smart Alert System</h3>', unsafe_allow_html=True)
        st.write("Set up automatic notifications when sensor readings exceed your defined thresholds")

        # Alert configuration section
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📧 Email Notifications")
            email_enabled = st.checkbox("Enable email alerts", key="email_alerts")
            if email_enabled:
                user_email = st.text_input("Your email address:", placeholder="example@email.com")

        with col2:
            st.subheader("📱 SMS Notifications")
            sms_enabled = st.checkbox("Enable SMS alerts", key="sms_alerts")
            if sms_enabled:
                user_phone = st.text_input("Your phone number:", placeholder="+1234567890")

        # Threshold configuration
        st.subheader("⚙ Alert Thresholds")

        # Create columns for different sensor types
        thresh_col1, thresh_col2, thresh_col3, thresh_col4 = st.columns(4)

        with thresh_col1:
            st.write("*CO2 Levels (ppm)*")
            co2_min = st.number_input("Min CO2:", value=400.0, step=50.0, key="co2_min")
            co2_max = st.number_input("Max CO2:", value=2000.0, step=50.0, key="co2_max")

        with thresh_col2:
            st.write("*Temperature (°C)*")
            temp_min = st.number_input("Min Temp:", value=15.0, step=1.0, key="temp_min")
            temp_max = st.number_input("Max Temp:", value=35.0, step=1.0, key="temp_max")

        with thresh_col3:
            st.write("*Humidity (%)*")
            hum_min = st.number_input("Min Humidity:", value=30.0, step=5.0, key="hum_min")
            hum_max = st.number_input("Max Humidity:", value=90.0, step=5.0, key="hum_max")

        with thresh_col4:
            st.write("*Battery (V)*")
            battery_min = st.number_input("Min Battery:", value=3.5, step=0.1, key="battery_min")

        # Check alerts button
        if st.button("🔍 Check Current Status", type="primary"):
            # Define thresholds based on user input
            thresholds = {
                'co2': {'min': co2_min, 'max': co2_max},
                'temperature': {'min': temp_min, 'max': temp_max},
                'humidity': {'min': hum_min, 'max': hum_max},
                'battery': {'min': battery_min}
            }

            # Check for alerts
            alerts = check_sensor_thresholds(df, thresholds)

            if alerts:
                st.error(f"🚨 {len(alerts)} Alert(s) Detected!")

                for alert in alerts:
                    severity_color = "🔴" if alert['severity'] == 'high' else "🟡"
                    st.warning(
                        f"{severity_color} *{alert['sensor']}*: {alert['message']} (Current: {alert['current_value']:.1f})")

                # Show notification status
                if email_enabled and 'user_email' in locals() and user_email:
                    st.success("📧 Email notification ready!")

                if sms_enabled and 'user_phone' in locals() and user_phone:
                    st.success("📱 SMS notification ready!")

            else:
                st.success("✅ All sensors are within normal ranges!")

else:
    st.error("No data available. Please check your data source or upload a file.")
