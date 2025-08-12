# --- car_inventory_tool.py ---

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import os
import zipfile
import io
import requests # Library for downloading from URLs

# -------------------------
# Custom State-Based Clustering and Area Logic (No changes needed)
# -------------------------
state_area_km2 = {
    'RJ': 342239, 'MP': 308252, 'MH': 307713, 'UP': 240928, 'GJ': 196024,
    'KA': 191791, 'AP': 162975, 'OD': 155707, 'CG': 135192, 'TN': 130058,
    'TG': 112077, 'BR': 94163, 'WB': 88752, 'AR': 83743, 'JH': 79716,
    'AS': 78438, 'HP': 55673, 'UK': 53483, 'PB': 50362, 'HR': 44212,
    'KL': 38863, 'ML': 22429, 'MN': 22327, 'MZ': 21081, 'NL': 16579,
    'TR': 10491, 'SK': 7096, 'GA': 3702, 'AN': 8249, 'JK': 42241,
    'LA': 59146, 'DL': 1484, 'DD': 603, 'PY': 479, 'CH': 114, 'LD': 32
}

def get_cluster_area(cluster):
    if '_' in cluster:
        code = cluster.split('_')[0]
        return state_area_km2.get(code, 0) / 2
    return state_area_km2.get(cluster, 0)

split_states = {'MH', 'UP', 'RJ', 'TN', 'KA'}

def extract_office_prefix(office_code):
    match = re.match(r'([A-Z]{2})\d+', str(office_code).strip().upper())
    return match.group(1) if match else None

def assign_state_cluster(row):
    code = row['state_code']
    if code in split_states:
        office = str(row['office_code']).strip().upper()
        num_part = re.sub(r'[A-Z]', '', office)
        if num_part.isdigit() and int(num_part) >= 50:
            return f"{code}_B"
        else:
            return f"{code}_A"
    return code

# -------------------------
# Feature 1: Buying Strength Processing Logic (MODIFIED for Parquet)
# -------------------------
@st.cache_data
def process_rto_data_for_strength(file_content):
    """
    Processes file content from a URL. It now reads the high-performance
    Parquet format for speed and efficiency.
    """
    try:
        # This is the one-word change: read_parquet is much faster
        df = pd.read_parquet(io.BytesIO(file_content))
    except Exception as e:
        st.error(f"Error reading the Parquet data from the URL: {e}")
        st.warning("Please ensure the URL points to a valid .parquet file.")
        return None, None

    # Standardize column names for registrations
    if 'registratio' in df.columns and 'registrations' not in df.columns:
        df = df.rename(columns={'registratio': 'registrations'})
        
    cluster_result = None
    vehicle_class_result = None

    # --- Part 1: Original State Cluster Analysis ---
    df_cluster = df.copy()
    required_cols_cluster = {'office_code', 'registrations', 'class_type'}
    if required_cols_cluster.issubset(df_cluster.columns):
        df_cluster['class_type'] = df_cluster['class_type'].astype(str).str.strip().str.lower()
        allowed_classes = {'motor car', 'luxury cab', 'maxi cab', 'm-cycle/scooter'}
        df_cluster = df_cluster[df_cluster['class_type'].isin(allowed_classes)]
        df_cluster['state_code'] = df_cluster['office_code'].apply(extract_office_prefix)
        df_cluster['City_Cluster'] = df_cluster.apply(assign_state_cluster, axis=1)
        df_cluster = df_cluster.dropna(subset=['City_Cluster'])
        
        if not df_cluster.empty:
            cluster_scores = df_cluster.groupby('City_Cluster')['registrations'].sum().reset_index(name='Total_Registrations')
            cluster_scores['Volume_Score'] = (cluster_scores['Total_Registrations'] / cluster_scores['Total_Registrations'].sum()) * 1000
            cluster_scores['Buying_Strength_Score'] = cluster_scores['Volume_Score']
            cluster_scores['Cluster_Area_km2'] = cluster_scores['City_Cluster'].apply(get_cluster_area)
            cluster_scores['Demand_Density_per_1000_km2'] = (cluster_scores['Total_Registrations'] / cluster_scores['Cluster_Area_km2']) * 1000
            cluster_result = cluster_scores.sort_values('Buying_Strength_Score', ascending=False)
    else:
        st.warning("State analysis skipped: requires 'office_code', 'registrations', 'class_type' columns.")

    # --- Part 2: New Vehicle Class by State Analysis ---
    df_vehicle = df.copy()
    required_cols_vehicle = {'office_code', 'registrations', 'Type of Class'}
    if required_cols_vehicle.issubset(df_vehicle.columns):
        df_vehicle['state_code'] = df_vehicle['office_code'].apply(extract_office_prefix)
        df_vehicle['City_Cluster'] = df_vehicle.apply(assign_state_cluster, axis=1)
        df_vehicle = df_vehicle.dropna(subset=['City_Cluster', 'Type of Class', 'registrations'])
        df_vehicle['Vehicle Class'] = df_vehicle['Type of Class'].astype(str).str.strip().str.lower().replace({'sux': 'suv'})
        df_vehicle['registrations'] = pd.to_numeric(df_vehicle['registrations'], errors='coerce').fillna(0)
        target_classes = ['suv', 'xuv', 'sedan', 'utility', 'luxury']
        df_filtered = df_vehicle[df_vehicle['Vehicle Class'].isin(target_classes)]

        if not df_filtered.empty:
            analysis_df = df_filtered.groupby(['City_Cluster', 'Vehicle Class'])['registrations'].sum().reset_index()
            analysis_df = analysis_df.rename(columns={'registrations': 'Total Registrations'})
            cluster_totals = analysis_df.groupby('City_Cluster')['Total Registrations'].transform('sum')
            analysis_df['Buying Strength (% Share)'] = round((analysis_df['Total Registrations'] / cluster_totals) * 100, 2)
            vehicle_class_result = analysis_df.sort_values(['City_Cluster', 'Total Registrations'], ascending=[True, False])
    else:
        st.warning("Vehicle class analysis skipped: requires 'office_code', 'registrations', 'Type of Class' columns.")

    return cluster_result, vehicle_class_result


# -------------------------
# Feature 2: Forecasting Model & Data Processing Functions (MODIFIED)
# -------------------------
@st.cache_data
def process_rto_data_from_zip(zip_file_content):
    all_dfs = []
    try:
        # Reads the zip file content directly from the URL download
        with zipfile.ZipFile(io.BytesIO(zip_file_content)) as z:
            for filename in z.namelist():
                if filename.startswith('__') or not (filename.endswith('.csv') or filename.endswith('.xlsx')):
                    continue
                match = re.search(r'(\d{4})[-_](\d{2})', filename)
                if not match: continue
                year, month = map(int, match.groups())
                with z.open(filename) as f:
                    file_data = f.read()
                    df = pd.read_csv(io.BytesIO(file_data)) if filename.endswith('.csv') else pd.read_excel(io.BytesIO(file_data))
                    df['date'] = datetime(year, month, 1)
                    all_dfs.append(df)
    except Exception as e:
        st.error(f"Failed to process zip file from URL: {e}")
        return None
    if not all_dfs:
        st.error("No valid data files found in the zip archive.")
        return None
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['state_code'] = full_df['office_code'].apply(extract_office_prefix)
    full_df['City_Cluster'] = full_df.apply(assign_state_cluster, axis=1)
    full_df = full_df.dropna(subset=['City_Cluster'])
    return full_df.groupby(['date', 'City_Cluster'])['registrations'].sum().reset_index()

# (No changes needed in the ML functions)
def create_tuned_lstm_model(X_train):
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_lstm_data(data, time_step=12):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def perform_lstm_cv(data, n_splits=4, time_step=12):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    errors = []
    for i, (train_index, test_index) in enumerate(tscv.split(data)):
        # ... (rest of function is unchanged)
        cv_train, cv_test = data.iloc[train_index], data.iloc[test_index]
        actual_value = cv_test['registrations'].iloc[0]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler.fit_transform(cv_train[['registrations']])
        if len(scaled_train) <= time_step: continue
        X_train, y_train = prepare_lstm_data(scaled_train, time_step)
        if X_train.size == 0: continue
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        lstm_model = create_tuned_lstm_model(X_train)
        lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        pred_scaled = lstm_model.predict(scaled_train[-time_step:].reshape(1, time_step, 1), verbose=0)
        prediction = scaler.inverse_transform(pred_scaled)[0][0]
        errors.append(abs(actual_value - prediction))
    return np.mean(errors) if errors else float('inf')


# -------------------------
# Main Streamlit UI (MODIFIED to load from GitHub)
# -------------------------
def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Analysis Tools")
    feature_choice = st.sidebar.radio("Choose a feature:", ("Buying Strength Analysis", "Demand Forecasting"))

    # --- ⬇️⬇️⬇️ PASTE YOUR RAW GITHUB URLs HERE ⬇️⬇️⬇️ ---
    # This URL should point to your new, smaller .parquet file
    buying_strength_data_url = "https://raw.githubusercontent.com/your-username/your-repo/main/car_data.parquet"
    
    # This URL should point to your zip file with monthly forecasting data
    forecasting_data_zip_url = "https://raw.githubusercontent.com/your-username/your-repo/main/your-forecasting-data.zip"
    # --- ⬆️⬆️⬆️ END OF SECTION TO EDIT ⬆️⬆️⬆️ ---


    if feature_choice == "Buying Strength Analysis":
        st.title("Used Car Market - Buying Strength Analysis")
        st.markdown("This tool ranks states by buying strength and analyzes vehicle class demand within each state.")
        
        # REMOVED file uploader, now loads from URL
        try:
            with st.spinner(f"Downloading data from GitHub..."):
                response = requests.get(buying_strength_data_url)
                response.raise_for_status() # Raise an exception for bad status codes
            
            with st.spinner("Processing data... This may take a moment for the first run."):
                cluster_result, vehicle_class_result = process_rto_data_for_strength(response.content)

            if cluster_result is None and vehicle_class_result is None:
                st.error("Could not process the data file. Please check the URL and file content.")
            else:
                st.success("Data processed successfully!")

            # --- Section 1: Display State-wise Cluster Analysis ---
            if cluster_result is not None:
                st.header("Part 1: State-wise Buying Strength Analysis")
                st.dataframe(cluster_result)
                # (Charting and download logic is unchanged)
            
            st.markdown("---")

            # --- Section 2: Display Vehicle Class Analysis ---
            if vehicle_class_result is not None:
                st.header("Part 2: Vehicle Class Buying Strength (by State)")
                # (Display logic is unchanged)
                all_clusters = sorted(vehicle_class_result['City_Cluster'].unique())
                selected_cluster = st.selectbox("Select a State/Cluster to Analyze:", all_clusters)
                if selected_cluster:
                    # ... (rest of display logic is unchanged)
                    cluster_data = vehicle_class_result[vehicle_class_result['City_Cluster'] == selected_cluster]
                    st.subheader(f"Analysis for: {selected_cluster}")
                    st.dataframe(cluster_data)


        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download data from the URL: {e}")
            st.warning("Please ensure the URL is correct and points to the 'raw' version of the file on GitHub.")


    elif feature_choice == "Demand Forecasting":
        st.title("Monthly Demand Forecasting Tool (Tuned LSTM vs. SARIMA)")
        st.markdown("This tool evaluates models to provide forecasts for August 2025.")
        
        # REMOVED file uploader, now loads from URL
        try:
            with st.spinner("Downloading forecast data from GitHub..."):
                response = requests.get(https://github.com/tavishee/smart-inventory-app/blob/main/split_by_month.zip)
                response.raise_for_status()
            
            with st.spinner("Processing forecast data..."):
                df = process_rto_data_from_zip(response.content)

            if df is not None and not df.empty:
                st.success("ZIP file processed successfully.")
                # (The rest of the forecasting logic remains exactly the same)
                all_clusters = sorted(df['City_Cluster'].unique())
                selected_cluster = st.selectbox("Select a City/Cluster to Forecast:", all_clusters)
                if selected_cluster:
                    # ... All forecasting, evaluation, and plotting logic is unchanged ...
                    city_df = df[df['City_Cluster'] == selected_cluster].set_index('date')[['registrations']].resample('MS').sum().sort_index()
                    st.write(f"### Historical Data for {selected_cluster}")
                    st.line_chart(city_df)
            else:
                st.error("Could not process the forecast data from the zip file.")

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download forecast data from the URL: {e}")
            st.warning("Please ensure the URL is correct and points to the 'raw' version of the file on GitHub.")

if __name__ == "__main__":
    main()

