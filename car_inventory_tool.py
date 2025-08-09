import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet
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
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        required_cols = {'office_name', 'office_code', 'registrations', 'class_type'}
        if not required_cols.issubset(set(df.columns)):
            st.error("Uploaded file must contain 'office_name', 'office_code', 'registrations', and 'class_type' columns")
            return None

        df['class_type'] = df['class_type'].astype(str).str.strip().str.lower()
        allowed_classes = {'motor car', 'luxury cab', 'maxi cab', 'm-cycle/scooter'}
        df = df[df['class_type'].isin(allowed_classes)]

        df['state_code'] = df['office_code'].apply(extract_office_prefix)
        df['City_Cluster'] = df.apply(assign_state_cluster, axis=1)
        df = df[df['City_Cluster'].notna()]

        if df.empty:
            st.warning("Filtered dataset is empty after processing. Please check input data.")
            return None

        cluster_scores = df.groupby('City_Cluster')['registrations'].sum().reset_index(name='Total_Registrations')
        cluster_scores['Volume_Score'] = (cluster_scores['Total_Registrations'] / cluster_scores['Total_Registrations'].sum()) * 1000
        cluster_scores['Buying_Strength_Score'] = cluster_scores['Volume_Score']
        cluster_scores['Cluster_Area_km2'] = cluster_scores['City_Cluster'].apply(get_cluster_area)
        cluster_scores['Demand_Density_per_1000_km2'] = (
            cluster_scores['Total_Registrations'] / cluster_scores['Cluster_Area_km2']
        ) * 1000

        return cluster_scores.sort_values('Buying_Strength_Score', ascending=False)

    except Exception as e:
        st.error(f"Error processing RTO data for Buying Strength: {str(e)}")
        return None

# -------------------------
# Feature 2: Forecasting Model & Data Processing Functions
# -------------------------
def create_lstm_model(X_train):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
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
                # Ignore macOS resource fork files and non-data files
                if filename.startswith('__') or not (filename.endswith('.csv') or filename.endswith('.xlsx')):
                    continue
                
                # Use a flexible regex to find YYYY-MM or YYYY_MM
                match = re.search(r'(\d{4})[-_](\d{2})', filename)
                if not match:
                    st.warning(f"Could not extract date from filename: `{filename}`. Skipping file. Expected format `YYYY-MM` or `YYYY_MM`.")
                    continue
                
                year, month = map(int, match.groups())
                file_date = datetime(year, month, 1)

                with z.open(filename) as f:
                    file_data = f.read()
                    if filename.endswith('.csv'):
                        df = pd.read_csv(io.BytesIO(file_data))
                    else:
                        df = pd.read_excel(io.BytesIO(file_data))
                    
                    df['date'] = file_date
                    all_dfs.append(df)
    
    except Exception as e:
        st.error(f"Failed to process the zip file: {e}")
        return None
    
    if not all_dfs:
        st.error("No valid data files were found inside the zip archive.")
        return None

    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # Apply clustering logic
    full_df['state_code'] = full_df['office_code'].apply(extract_office_prefix)
    full_df['City_Cluster'] = full_df.apply(assign_state_cluster, axis=1)
    full_df = full_df[full_df['City_Cluster'].notna()]

    # Aggregate to monthly registrations per cluster
    time_series_df = full_df.groupby(['date', 'City_Cluster'])['registrations'].sum().reset_index()
    return time_series_df

# -------------------------
# Main Streamlit UI
# -------------------------
def main():
    st.set_page_config(layout="wide")
    
    st.sidebar.title("🛠️ Analysis Tools")
    feature_choice = st.sidebar.radio(
        "Choose a feature:",
        ("Buying Strength Analysis", "Demand Forecasting")
    )

    if feature_choice == "Buying Strength Analysis":
        st.title("🚗 Used Car Market - State-wise Buying Strength Analysis")
        st.markdown("This tool ranks Indian states by **used car buying strength** using a single data file. It analyzes total registrations and demand density.")
        
        with st.expander("📁 Upload RTO Data File", expanded=True):
            uploaded_file = st.file_uploader("Upload a single vehicle registration data file (CSV/Excel)", type=['csv', 'xlsx'])
            if uploaded_file:
                result = process_rto_data_for_strength(uploaded_file)
                if result is not None:
                    # ... (rest of the buying strength UI code is unchanged)
                    st.success("✅ Processed successfully.")
                    st.dataframe(result)

                    fig = go.Figure(data=[go.Bar(y=result['City_Cluster'], x=result['Buying_Strength_Score'], name='Buying Strength', orientation='h', marker_color='steelblue')])
                    fig.update_layout(title='📊 Buying Strength Score by State Cluster', height=800)
                    st.plotly_chart(fig, use_container_width=True)

                    fig_density = go.Figure(data=[go.Bar(y=result['City_Cluster'], x=result['Demand_Density_per_1000_km2'], name='Demand Density', orientation='h', marker_color='darkorange')])
                    fig_density.update_layout(title='🌐 Demand Density per 1000 km² by State Cluster', height=800)
                    st.plotly_chart(fig_density, use_container_width=True)

                    st.download_button("📥 Download Results as CSV", result.to_csv(index=False), f"buying_strength_{datetime.now().strftime('%Y%m%d')}.csv", 'text/csv')

    elif feature_choice == "Demand Forecasting":
        st.title("📈 Monthly Demand Forecasting Tool")
        st.markdown("""
        Forecast future RTO registrations using LSTM and Prophet models. This tool requires a **single ZIP file** containing all your monthly data files.
        - **Filename format inside the ZIP must be `prefix_YYYY-MM.csv` or `prefix_YYYY_MM.xlsx`**.
        - The tool validates models on May 2024 data and then forecasts for August 2025.
        """)

        with st.expander("📁 Upload Monthly RTO Data ZIP File", expanded=True):
            uploaded_zip_file = st.file_uploader("Upload a single ZIP file containing all monthly data", type=['zip'])
            
            if uploaded_zip_file:
                df = process_rto_data_from_zip(uploaded_zip_file)
                
                if df is not None and not df.empty:
                    st.success("✅ ZIP file processed successfully.")
                    
                    all_clusters = sorted(df['City_Cluster'].unique())
                    selected_cluster = st.selectbox("Select a City/Cluster to Forecast:", all_clusters)
                    
                    city_df = df[df['City_Cluster'] == selected_cluster].copy()
                    city_df.set_index('date', inplace=True)
                    city_df = city_df[['registrations']].resample('MS').sum().sort_index()

                    st.write(f"### Historical Registration Data for {selected_cluster}")
                    st.line_chart(city_df)

                    st.header(f"Forecast vs. Actual for May 2024 in {selected_cluster}")
                    train_df = city_df[city_df.index < '2024-05-01']
                    may_actual = city_df[city_df.index == '2024-05-01']

                    if not may_actual.empty and len(train_df) > 12:
                        # --- Run Forecasting Logic (this part remains the same) ---
                        actual_may_value = may_actual['registrations'].iloc[0]
                        time_step = 12

                        # LSTM
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        scaled_train_data = scaler.fit_transform(train_df)
                        last_12_months = scaled_train_data[-time_step:]
                        X_pred = last_12_months.reshape(1, time_step, 1)
                        X_train, y_train = prepare_lstm_data(scaled_train_data, time_step)
                        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                        lstm_model = create_lstm_model(X_train)
                        lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                        lstm_may_pred_scaled = lstm_model.predict(X_pred)
                        lstm_may_pred = scaler.inverse_transform(lstm_may_pred_scaled)[0][0]

                        # Prophet
                        prophet_df = train_df.reset_index().rename(columns={'date': 'ds', 'registrations': 'y'})
                        prophet_model = Prophet()
                        prophet_model.fit(prophet_df)
                        future_may = prophet_model.make_future_dataframe(periods=1, freq='MS')
                        prophet_may_forecast = prophet_model.predict(future_may)
                        prophet_may_pred = prophet_may_forecast[prophet_may_forecast['ds'] == '2024-05-01']['yhat'].iloc[0]

                        # UI Display for May comparison
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("#### LSTM Performance (May 2024)")
                            st.metric("Predicted", f"{int(lstm_may_pred):,}", f"Actual: {int(actual_may_value):,}")
                            lstm_error = abs(lstm_may_pred - actual_may_value)
                            st.metric("Absolute Error", f"{int(lstm_error):,}")
                        with col2:
                            st.write("#### Prophet Performance (May 2024)")
                            st.metric("Predicted", f"{int(prophet_may_pred):,}", f"Actual: {int(actual_may_value):,}")
                            prophet_error = abs(prophet_may_pred - actual_may_value)
                            st.metric("Absolute Error", f"{int(prophet_error):,}")

                        # August 2025 Forecast
                        st.header(f"Final Forecast for August 2025 in {selected_cluster}")
                        # ... (rest of the forecasting and display logic is unchanged)
                        months_to_forecast = (2025 - 2024) * 12 + (8 - 5)
                        
                        full_df_ts = city_df.copy()
                        scaled_full_data = scaler.fit_transform(full_df_ts)
                        temp_input = list(scaled_full_data.flatten())
                        lstm_output = []
                        for _ in range(months_to_forecast):
                            x_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
                            yhat = lstm_model.predict(x_input, verbose=0)
                            temp_input.append(yhat[0,0])
                            lstm_output.append(yhat[0,0])
                        lstm_aug_pred = scaler.inverse_transform(np.array([[lstm_output[-1]]]))[0][0]

                        prophet_df_full = full_df_ts.reset_index().rename(columns={'date': 'ds', 'registrations': 'y'})
                        prophet_model_full = Prophet()
                        prophet_model_full.fit(prophet_df_full)
                        future_aug = prophet_model_full.make_future_dataframe(periods=months_to_forecast, freq='MS')
                        prophet_aug_forecast = prophet_model_full.predict(future_aug)
                        prophet_aug_pred = prophet_aug_forecast[prophet_aug_forecast['ds'] == '2025-08-01']['yhat'].iloc[0]

                        cluster_area = get_cluster_area(selected_cluster)
                        lstm_demand_density = (lstm_aug_pred / cluster_area) * 1000 if cluster_area > 0 else 0
                        prophet_demand_density = (prophet_aug_pred / cluster_area) * 1000 if cluster_area > 0 else 0

                        col3, col4 = st.columns(2)
                        with col3:
                            st.write("#### LSTM Forecast (August 2025)")
                            st.metric("Predicted Volume", f"{int(lstm_aug_pred):,}")
                            st.metric("Predicted Density / 1000 km²", f"{lstm_demand_density:.2f}")
                        with col4:
                            st.write("#### Prophet Forecast (August 2025)")
                            st.metric("Predicted Volume", f"{int(prophet_aug_pred):,}")
                            st.metric("Predicted Density / 1000 km²", f"{prophet_demand_density:.2f}")
                        
                        fig, ax = plt.subplots()
                        ax.plot(city_df.index, city_df['registrations'], label='Historical Actual', marker='o', linestyle='-')
                        ax.axvline(x=pd.to_datetime('2024-05-01'), color='gray', linestyle='--', label='May Prediction Point')
                        ax.scatter(pd.to_datetime('2024-05-01'), lstm_may_pred, color='red', s=100, zorder=5, label=f'LSTM Pred: {int(lstm_may_pred):,}')
                        ax.scatter(pd.to_datetime('2024-05-01'), prophet_may_pred, color='green', s=100, zorder=5, label=f'Prophet Pred: {int(prophet_may_pred):,}')
                        ax.scatter(pd.to_datetime('2024-05-01'), actual_may_value, color='blue', s=100, zorder=5, label=f'Actual: {int(actual_may_value):,}')
                        plt.title(f'May 2024: Actual vs. Predicted for {selected_cluster}')
                        plt.legend(); plt.grid(True)
                        st.pyplot(fig)
                    
                    else:
                        st.warning("Not enough data to run forecast for the selected cluster. Need at least 13 months of historical data and a valid value for May 2024.")

if __name__ == "__main__":
    main()



