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
# Feature 1: Buying Strength Processing Logic (MODIFIED for State-by-State Vehicle Class)
# -------------------------
def process_rto_data_for_strength(uploaded_file):
    """
    Processes the uploaded data file to calculate buying strength of vehicle classes
    on a per-state-cluster basis.
    Returns a single, detailed DataFrame.
    """
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading the uploaded file: {e}")
        return None

    # --- Standardize Column Names and Data ---
    if 'registratio' in df.columns and 'registrations' not in df.columns:
        df = df.rename(columns={'registratio': 'registrations'})

    required_cols = {'office_code', 'registrations', 'Type of Class'}
    if not required_cols.issubset(df.columns):
        st.error("Uploaded file must contain 'office_code', 'registrations', and 'Type of Class' columns for this analysis.")
        return None

    # --- Data Cleaning and Feature Engineering ---
    df['state_code'] = df['office_code'].apply(extract_office_prefix)
    df['City_Cluster'] = df.apply(assign_state_cluster, axis=1)

    df = df.dropna(subset=['City_Cluster', 'Type of Class', 'registrations'])
    df['Vehicle Class'] = df['Type of Class'].astype(str).str.strip().str.lower()
    df['Vehicle Class'] = df['Vehicle Class'].replace({'sux': 'suv'})
    df['registrations'] = pd.to_numeric(df['registrations'], errors='coerce').fillna(0)

    target_classes = ['suv', 'xuv', 'sedan', 'utility', 'luxury']
    df_filtered = df[df['Vehicle Class'].isin(target_classes)]

    if df_filtered.empty:
        st.warning("No data found for the target vehicle classes (suv, xuv, sedan, utility, luxury).")
        return None

    # --- Perform Grouping by Cluster and Vehicle Class ---
    analysis_df = df_filtered.groupby(['City_Cluster', 'Vehicle Class'])['registrations'].sum().reset_index()
    analysis_df = analysis_df.rename(columns={'registrations': 'Total Registrations'})
    
    # Calculate Buying Strength (Market Share %) within each cluster
    cluster_totals = analysis_df.groupby('City_Cluster')['Total Registrations'].transform('sum')
    analysis_df['Buying Strength (% Share in Cluster)'] = round((analysis_df['Total Registrations'] / cluster_totals) * 100, 2)

    return analysis_df.sort_values(['City_Cluster', 'Total Registrations'], ascending=[True, False])


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
    st.write(f"--- Running {n_splits}-Fold Cross-Validation for Tuned LSTM ---")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    errors = []
    progress_bar = st.progress(0)
    for i, (train_index, test_index) in enumerate(tscv.split(data)):
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
        progress_bar.progress((i + 1) / n_splits)
    progress_bar.empty()
    return np.mean(errors) if errors else float('inf')


# -------------------------
# Main Streamlit UI (MODIFIED)
# -------------------------
def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("🛠️ Analysis Tools")
    feature_choice = st.sidebar.radio("Choose a feature:", ("Buying Strength Analysis", "Demand Forecasting"))

    if feature_choice == "Buying Strength Analysis":
        st.title("🚗 Vehicle Buying Strength by State and Class")
        st.markdown("This tool analyzes buying strength for specific vehicle classes within each state/cluster. Upload a file to begin.")
        uploaded_file = st.file_uploader("Upload your vehicle registration data file (CSV/Excel)", type=['csv', 'xlsx'])

        if uploaded_file:
            analysis_df = process_rto_data_for_strength(uploaded_file)

            if analysis_df is not None and not analysis_df.empty:
                st.success("✅ Data processed successfully. Select a state/cluster to view the analysis.")

                all_clusters = sorted(analysis_df['City_Cluster'].unique())
                selected_cluster = st.selectbox("Select a State/Cluster to Analyze:", all_clusters)

                if selected_cluster:
                    st.header(f"Vehicle Class Analysis for: {selected_cluster}")

                    cluster_data = analysis_df[analysis_df['City_Cluster'] == selected_cluster]

                    # Display Bar chart for total registrations
                    bar_fig = go.Figure(data=[go.Bar(
                        x=cluster_data['Vehicle Class'],
                        y=cluster_data['Total Registrations'],
                        text=cluster_data['Total Registrations'],
                        textposition='auto',
                        marker_color='royalblue'
                    )])
                    bar_fig.update_layout(
                        title=f'📊 Total Registrations by Vehicle Class in {selected_cluster}',
                        xaxis_title='Vehicle Class',
                        yaxis_title='Total Registrations',
                        xaxis={'categoryorder':'total descending'}
                    )
                    st.plotly_chart(bar_fig, use_container_width=True)
                    
                    # Display Pie chart for market share
                    pie_fig = go.Figure(data=[go.Pie(
                        labels=cluster_data['Vehicle Class'],
                        values=cluster_data['Total Registrations'],
                        textinfo='percent+label',
                        hole=.3
                    )])
                    pie_fig.update_layout(
                        title=f'📈 Vehicle Class Market Share in {selected_cluster}'
                    )
                    st.plotly_chart(pie_fig, use_container_width=True)

                    # Display data table
                    st.subheader("Detailed Data")
                    st.dataframe(cluster_data)

                    st.download_button(
                        label=f"📥 Download Data for {selected_cluster}",
                        data=cluster_data.to_csv(index=False),
                        file_name=f"buying_strength_{selected_cluster}.csv",
                        mime='text/csv',
                        key='download_filtered'
                    )
                
                st.markdown("---")
                st.download_button(
                    label="📥 Download Full Analysis Data (All Clusters)",
                    data=analysis_df.to_csv(index=False),
                    file_name="full_vehicle_class_analysis.csv",
                    mime='text/csv',
                    key='download_full'
                )

    elif feature_choice == "Demand Forecasting":
        # (The forecasting feature remains unchanged)
        st.title("📈 Monthly Demand Forecasting Tool (Tuned LSTM vs. SARIMA)")
        st.markdown("This tool evaluates Tuned LSTM and SARIMA models to provide forecasts for August 2025.")
        uploaded_zip_file = st.file_uploader("Upload a single ZIP file with monthly data (`prefix_YYYY-MM.csv`)", type=['zip'])
        if uploaded_zip_file:
            df = process_rto_data_from_zip(uploaded_zip_file)
            if df is not None and not df.empty:
                st.success("✅ ZIP file processed successfully.")
                all_clusters = sorted(df['City_Cluster'].unique())
                selected_cluster = st.selectbox("Select a City/Cluster to Forecast:", all_clusters)
                city_df = df[df['City_Cluster'] == selected_cluster].set_index('date')[['registrations']].resample('MS').sum().sort_index()

                st.write(f"### Historical Data for {selected_cluster}")
                st.line_chart(city_df)

                st.header(f"Model Performance Evaluation")
                train_df = city_df[city_df.index < '2024-05-01']
                may_actual = city_df[city_df.index == '2024-05-01']

                if not may_actual.empty and len(train_df) > 12:
                    actual_may_value = may_actual['registrations'].iloc[0]
                    time_step = 12
                    
                    with st.spinner("Evaluating models... This may take a moment."):
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        scaled_train = scaler.fit_transform(train_df)
                        X_train, y_train = prepare_lstm_data(scaled_train, time_step)
                        
                        lstm_combined_error = float('inf')
                        lstm_may_pred = 0

                        if X_train.size > 0:
                            X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                            lstm_model = create_tuned_lstm_model(X_train_reshaped)
                            lstm_model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, verbose=0)
                            lstm_may_pred = scaler.inverse_transform(lstm_model.predict(scaled_train[-time_step:].reshape(1, time_step, 1), verbose=0))[0][0]
                            lstm_may_error = abs(lstm_may_pred - actual_may_value)
                            lstm_cv_error = perform_lstm_cv(train_df)
                            lstm_combined_error = (0.7 * lstm_cv_error) + (0.3 * lstm_may_error)
                        else:
                            st.warning("Skipping LSTM evaluation due to insufficient data.")

                        sarima_order = (1, 1, 1)
                        seasonal_order = (1, 1, 1, 12)
                        sarima_model = SARIMAX(train_df['registrations'], order=sarima_order, seasonal_order=seasonal_order).fit(disp=False)
                        sarima_may_pred = sarima_model.predict(start=len(train_df), end=len(train_df)).iloc[0]
                        sarima_may_error = abs(sarima_may_pred - actual_may_value)

                    st.subheader("May 2024 Prediction vs. Actual")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("#### Tuned LSTM Performance")
                        st.metric("Predicted (May 2024)", f"{int(lstm_may_pred):,}", f"Actual: {int(actual_may_value):,}")
                        st.metric("Final Combined Error", f"{int(lstm_combined_error):,}")
                    with col2:
                        st.write("#### SARIMA Performance")
                        st.metric("Predicted (May 2024)", f"{int(sarima_may_pred):,}", f"Actual: {int(actual_may_value):,}")
                        st.metric("Absolute Error", f"{int(sarima_may_error):,}")

                    st.markdown("---")
                    st.header(f"Final Forecast for August 2025")
                    months_to_forecast = (datetime(2025, 8, 1) - city_df.index.max()).days // 30
                    full_df_ts = city_df.copy()

                    with st.spinner("Retraining & forecasting with Tuned LSTM..."):
                        scaled_full = scaler.fit_transform(full_df_ts)
                        X_full, y_full = prepare_lstm_data(scaled_full, time_step)
                        if len(X_full) > 0:
                            X_full_reshaped = X_full.reshape(X_full.shape[0], X_full.shape[1], 1)
                            lstm_full_model = create_tuned_lstm_model(X_full_reshaped)
                            lstm_full_model.fit(X_full_reshaped, y_full, epochs=50, batch_size=32, verbose=0)
                            temp_input = list(scaled_full[-time_step:].flatten())
                            lstm_output = []
                            for _ in range(months_to_forecast):
                                yhat = lstm_full_model.predict(np.array(temp_input[-time_step:]).reshape(1, time_step, 1), verbose=0)
                                temp_input.append(yhat[0,0])
                                lstm_output.append(yhat[0,0])
                            lstm_aug_pred = scaler.inverse_transform(np.array([[lstm_output[-1]]]))[0][0]
                        else:
                            lstm_aug_pred = 0

                    with st.spinner("Retraining & forecasting with SARIMA..."):
                        sarima_model_full = SARIMAX(full_df_ts['registrations'], order=sarima_order, seasonal_order=seasonal_order).fit(disp=False)
                        sarima_aug_pred = sarima_model_full.get_forecast(steps=months_to_forecast).predicted_mean.iloc[-1]
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        st.write("#### Tuned LSTM Forecast (August 2025)")
                        st.metric("Predicted Volume", f"{int(lstm_aug_pred):,}")
                        cluster_area = get_cluster_area(selected_cluster)
                        st.metric("Predicted Density / 1000 km²", f"{(lstm_aug_pred / cluster_area) * 1000:.2f}" if cluster_area > 0 else "N/A")
                    with col4:
                        st.write("#### SARIMA Forecast (August 2025)")
                        st.metric("Predicted Volume", f"{int(sarima_aug_pred):,}")
                        cluster_area = get_cluster_area(selected_cluster)
                        st.metric("Predicted Density / 1000 km²", f"{(sarima_aug_pred / cluster_area) * 1000:.2f}" if cluster_area > 0 else "N/A")

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
