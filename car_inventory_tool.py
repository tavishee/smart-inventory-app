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

# -------------------------
# Custom State-Based Clustering and Area Logic
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
# Feature 1: Buying Strength Processing Logic
# -------------------------
def process_rto_data_for_strength(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        required_cols = {'office_name', 'office_code', 'registrations', 'class_type'}
        if not required_cols.issubset(df.columns):
            st.error("Uploaded file must contain 'office_name', 'office_code', 'registrations', and 'class_type' columns")
            return None
        df['class_type'] = df['class_type'].astype(str).str.strip().str.lower()
        allowed_classes = {'motor car', 'luxury cab', 'maxi cab', 'm-cycle/scooter'}
        df = df[df['class_type'].isin(allowed_classes)]
        df['state_code'] = df['office_code'].apply(extract_office_prefix)
        df['City_Cluster'] = df.apply(assign_state_cluster, axis=1)
        df = df.dropna(subset=['City_Cluster'])
        if df.empty:
            st.warning("Filtered dataset is empty. Please check input data.")
            return None
        cluster_scores = df.groupby('City_Cluster')['registrations'].sum().reset_index(name='Total_Registrations')
        cluster_scores['Volume_Score'] = (cluster_scores['Total_Registrations'] / cluster_scores['Total_Registrations'].sum()) * 1000
        cluster_scores['Buying_Strength_Score'] = cluster_scores['Volume_Score']
        cluster_scores['Cluster_Area_km2'] = cluster_scores['City_Cluster'].apply(get_cluster_area)
        cluster_scores['Demand_Density_per_1000_km2'] = (cluster_scores['Total_Registrations'] / cluster_scores['Cluster_Area_km2']) * 1000
        return cluster_scores.sort_values('Buying_Strength_Score', ascending=False)
    except Exception as e:
        st.error(f"Error processing data for Buying Strength: {e}")
        return None

# -------------------------
# Feature 2: Forecasting Model & Data Processing Functions
# -------------------------
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

def process_rto_data_from_zip(uploaded_zip_file):
    all_dfs = []
    try:
        with zipfile.ZipFile(uploaded_zip_file) as z:
            for filename in z.namelist():
                if filename.startswith('__') or not (filename.endswith('.csv') or filename.endswith('.xlsx')):
                    continue
                match = re.search(r'(\d{4})[-_](\d{2})', filename)
                if not match:
                    st.warning(f"Could not parse date from: `{filename}`. Skipping.")
                    continue
                year, month = map(int, match.groups())
                with z.open(filename) as f:
                    file_data = f.read()
                    df = pd.read_csv(io.BytesIO(file_data)) if filename.endswith('.csv') else pd.read_excel(io.BytesIO(file_data))
                    df['date'] = datetime(year, month, 1)
                    all_dfs.append(df)
    except Exception as e:
        st.error(f"Failed to process zip file: {e}")
        return None
    if not all_dfs:
        st.error("No valid data files found in zip archive.")
        return None
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['state_code'] = full_df['office_code'].apply(extract_office_prefix)
    full_df['City_Cluster'] = full_df.apply(assign_state_cluster, axis=1)
    full_df = full_df.dropna(subset=['City_Cluster'])
    return full_df.groupby(['date', 'City_Cluster'])['registrations'].sum().reset_index()

def perform_lstm_cv(data, n_splits=4, time_step=12):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    errors = []
    for train_index, test_index in tscv.split(data):
        cv_train, cv_test = data.iloc[train_index], data.iloc[test_index]
        actual_value = cv_test['registrations'].iloc[0]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler.fit_transform(cv_train[['registrations']])
        if len(scaled_train) <= time_step: continue
        X_train, y_train = prepare_lstm_data(scaled_train, time_step)
        if X_train.size == 0: continue
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        lstm_model = create_tuned_lstm_model(X_train)
