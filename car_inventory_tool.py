import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout  # Dropout is added for tuning
from statsmodels.tsa.statespace.sarimax import SARIMAX
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

# THIS IS THE TUNED LSTM MODEL FUNCTION
def create_tuned_lstm_model(X_train):
    model = Sequential()
    # Layer 1: More units and return_sequences for stacking
    model.add(LSTM(100, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=True))
    # Dropout layer to prevent overfitting
    model.add(Dropout(0.2))
    # Layer 2: Learns from the first layer's patterns
    model.add(LSTM(50, activation='relu'))
    # Another dropout layer
    model.add(Dropout(0.2))
    # Final output layer
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

# -------------------------
# Main Streamlit UI
# -------------------------
def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("🛠️ Analysis Tools")
    feature_choice = st.sidebar.radio("Choose a feature:", ("Buying Strength Analysis", "Demand Forecasting"))

    if feature_choice == "Buying Strength Analysis":
        st.title("🚗 Used Car Market - State-wise Buying Strength Analysis")
        st.markdown("This tool ranks states by buying strength using a single data file.")
        uploaded_file = st.file_uploader("Upload a single vehicle registration data file (CSV/Excel)", type=['csv', 'xlsx'])
        if uploaded_file:
            result = process_rto_data_for_strength(uploaded_file)
            if result is not None:
                st.success("✅ Processed successfully.")
                st.dataframe(result)
                fig = go.Figure(data=[go.Bar(y=result['City_Cluster'], x=result['Buying_Strength_Score'], name='Buying Strength', orientation='h', marker_color='steelblue')])
                fig.update_layout(title='📊 Buying Strength Score by State Cluster', height=800)
                st.plotly_chart(fig, use_container_width=True)
                fig_density = go.Figure(data=[go.Bar(y=result['City_Cluster'], x=result['Demand_Density_per_1000_km2'], name='Demand Density', orientation='h', marker_color='darkorange')])
                fig_density.update_layout(title='🌐 Demand Density per 1000 km² by State Cluster', height=800)
                st.plotly_chart(fig_density, use_container_width=True)
                st.download_button("📥 Download Results", result.to_csv(index=False), f"buying_strength.csv", 'text/csv')

    elif feature_choice == "Demand Forecasting":
        st.title("📈 Monthly Demand Forecasting Tool (Tuned LSTM vs. SARIMA)")
        st.markdown("Forecast future demand using a Tuned LSTM and SARIMA. Requires a **single ZIP file** with monthly data (`prefix_YYYY-MM.csv`).")
        uploaded_zip_file = st.file_uploader("Upload a single ZIP file", type=['zip'])
        if uploaded_zip_file:
            df = process_rto_data_from_zip(uploaded_zip_file)
            if df is not None and not df.empty:
                st.success("✅ ZIP file processed successfully.")
                all_clusters = sorted(df['City_Cluster'].unique())
                selected_cluster = st.selectbox("Select a City/Cluster to Forecast:", all_clusters)
                city_df = df[df['City_Cluster'] == selected_cluster].set_index('date')[['registrations']].resample('MS').sum().sort_index()

                st.write(f"### Historical Data for {selected_cluster}")
                st.line_chart(city_df)

                st.header(f"Forecast vs. Actual for May 2024")
                train_df = city_df[city_df.index < '2024-05-01']
                may_actual = city_df[city_df.index == '2024-05-01']

                if not may_actual.empty and len(train_df) > 12:
                    actual_may_value = may_actual['registrations'].iloc[0]
                    time_step = 12
                    
                    # --- May 2024 Validation ---
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_train = scaler.fit_transform(train_df)
                    X_train, y_train = prepare_lstm_data(scaled_train, time_step)
                    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                    
                    with st.spinner("Fitting Tuned LSTM for May validation..."):
                        # USING THE TUNED LSTM MODEL
                        lstm_model = create_tuned_lstm_model(X_train)
                        lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                        lstm_may_pred = scaler.inverse_transform(lstm_model.predict(scaled_train[-time_step:].reshape(1, time_step, 1), verbose=0))[0][0]

                    with st.spinner("Fitting SARIMA for May validation..."):
                        # Using original SARIMA as requested
                        sarima_order = (1, 1, 1)
                        seasonal_order = (1, 1, 1, 12)
                        sarima_model = SARIMAX(train_df['registrations'], order=sarima_order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                        sarima_fit = sarima_model.fit(disp=False)
                        sarima_may_pred = sarima_fit.predict(start=len(train_df), end=len(train_df), dynamic=False).iloc[0]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("#### Tuned LSTM Performance (May 2024)")
                        st.metric("Predicted", f"{int(lstm_may_pred):,}", f"Actual: {int(actual_may_value):,}")
                        st.metric("Absolute Error", f"{abs(lstm_may_pred - actual_may_value):,.0f}")
                    with col2:
                        st.write("#### SARIMA Performance (May 2024)")
                        st.metric("Predicted", f"{int(sarima_may_pred):,}", f"Actual: {int(actual_may_value):,}")
                        st.metric("Absolute Error", f"{abs(sarima_may_pred - actual_may_value):,.0f}")

                    # --- August 2025 Forecasting ---
                    st.header(f"Final Forecast for August 2025")
                    months_to_forecast = (datetime(2025, 8, 1) - city_df.index.max()).days // 30
                    full_df_ts = city_df.copy()

                    # 1. Retrain and Forecast with TUNED LSTM
                    with st.spinner("Retraining & forecasting with Tuned LSTM..."):
                        scaled_full = scaler.fit_transform(full_df_ts)
                        X_full, y_full = prepare_lstm_data(scaled_full, time_step)
                        if len(X_full) > 0:
                            X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], 1)
                            # USING THE TUNED LSTM MODEL
                            lstm_full_model = create_tuned_lstm_model(X_full)
                            lstm_full_model.fit(X_full, y_full, epochs=50, batch_size=32, verbose=0)
                            temp_input = list(scaled_full[-time_step:].flatten())
                            lstm_output = []
                            for _ in range(months_to_forecast):
                                yhat = lstm_full_model.predict(np.array(temp_input[-time_step:]).reshape(1, time_step, 1), verbose=0)
                                temp_input.append(yhat[0,0])
                                lstm_output.append(yhat[0,0])
                            lstm_aug_pred = scaler.inverse_transform(np.array([[lstm_output[-1]]]))[0][0]
                        else:
                            lstm_aug_pred = 0

                    # 2. Retrain and Forecast with ORIGINAL SARIMA
                    with st.spinner("Retraining & forecasting with SARIMA..."):
                        sarima_model_full = SARIMAX(full_df_ts['registrations'], order=sarima_order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                        sarima_fit_full = sarima_model_full.fit(disp=False)
                        sarima_aug_pred = sarima_fit_full.get_forecast(steps=months_to_forecast).predicted_mean.iloc[-1]
                    
                    # --- Display Final Forecasts ---
                    cluster_area = get_cluster_area(selected_cluster)
                    col3, col4 = st.columns(2)
                    with col3:
                        st.write("#### Tuned LSTM Forecast (August 2025)")
                        st.metric("Predicted Volume", f"{int(lstm_aug_pred):,}")
                        st.metric("Predicted Density / 1000 km²", f"{(lstm_aug_pred / cluster_area) * 1000:.2f}" if cluster_area > 0 else "N/A")
                    with col4:
                        st.write("#### SARIMA Forecast (August 2025)")
                        st.metric("Predicted Volume", f"{int(sarima_aug_pred):,}")
                        st.metric("Predicted Density / 1000 km²", f"{(sarima_aug_pred / cluster_area) * 1000:.2f}" if cluster_area > 0 else "N/A")

                    # --- Visualization ---
                    st.subheader("Visual Comparison of May 2024 Validation")
                    fig, ax = plt.subplots()
                    ax.plot(city_df.index, city_df['registrations'], label='Historical', marker='o', linestyle='-')
                    ax.axvline(x=pd.to_datetime('2024-05-01'), color='gray', linestyle='--')
                    ax.scatter(pd.to_datetime('2024-05-01'), lstm_may_pred, color='red', s=100, zorder=5, label=f'Tuned LSTM: {int(lstm_may_pred):,}')
                    ax.scatter(pd.to_datetime('2024-05-01'), sarima_may_pred, color='purple', s=100, zorder=5, label=f'SARIMA: {int(sarima_may_pred):,}')
                    ax.scatter(pd.to_datetime('2024-05-01'), actual_may_value, color='blue', s=100, zorder=5, label=f'Actual: {int(actual_may_value):,}')
                    plt.title(f'May 2024 Validation for {selected_cluster}')
                    plt.legend(); plt.grid(True)
                    st.pyplot(fig)
                else:
                    st.warning("Not enough data to run forecast. Requires >12 months of data and a valid value for May 2024.")

if __name__ == "__main__":
    main()
