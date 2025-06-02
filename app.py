import streamlit as st
import os
import pandas as pd
from datetime import datetime

st.title("🔍 File System Diagnostic")

# Check current working directory
st.write(f"**Current directory:** {os.getcwd()}")

# List all items in root directory
st.subheader("Root Directory Contents:")
try:
    root_files = os.listdir('.')
    for item in sorted(root_files):
        if os.path.isdir(item):
            st.write(f"📁 {item}/")
        else:
            st.write(f"📄 {item}")
except Exception as e:
    st.error(f"Cannot list root directory: {e}")

# Check for attached_assets folder
st.subheader("Checking attached_assets folder:")
paths_to_check = ['./attached_assets', 'attached_assets', 'attached-assets']

for path in paths_to_check:
    if os.path.exists(path):
        st.success(f"✅ Found: {path}")
        try:
            files = os.listdir(path)
            st.write(f"Contents: {files}")
            
            # Try to read Experiment7.txt
            exp_file = os.path.join(path, 'Experiment7.txt')
            if os.path.exists(exp_file):
                st.write("**Testing Experiment7.txt:**")
                with open(exp_file, 'r') as f:
                    lines = f.readlines()
                st.write(f"- Total lines: {len(lines)}")
                if len(lines) > 1:
                    st.write(f"- Sample line: {lines[1][:100]}...")
                    # Parse sample data
                    parts = lines[1].strip().split(',')
                    if len(parts) >= 20:
                        st.write(f"- Timestamp: {parts[0]}")
                        st.write(f"- CO2 values: {parts[1:5]}")
                break
        except Exception as e:
            st.error(f"Error reading from {path}: {e}")
    else:
        st.warning(f"❌ Not found: {path}")

# Test actual data loading
st.subheader("Test Real Data Loading:")
if st.button("Load Test Data"):
    # Try to find and load data
    for path in ['./attached_assets', 'attached_assets']:
        exp_file = os.path.join(path, 'Experiment7.txt')
        if os.path.exists(exp_file):
            try:
                with open(exp_file, 'r') as f:
                    content = f.read()
                lines = content.strip().split('\n')
                
                data_rows = []
                for line in lines[1:6]:  # Test first 5 data lines
                    parts = line.strip().split(',')
                    if len(parts) >= 20:
                        time_str = parts[0].strip()
                        timestamp = datetime.strptime(time_str, '%H:%M:%S %d/%m/%Y')
                        co2_a = float(parts[1])
                        temp_a = float(parts[2])
                        data_rows.append([timestamp, co2_a, temp_a])
                
                df = pd.DataFrame(data_rows, columns=['timestamp', 'CO2_A', 'Temp_A'])
                st.write("**Successfully loaded real data:**")
                st.dataframe(df)
                
                # Show a simple plot
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['CO2_A'], name='CO2_A'))
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error processing data: {e}")
            break
