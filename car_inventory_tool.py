import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(layout="wide")

# State capital mappings
STATE_CAPITAL_CLUSTERS = {
    'Andhra Pradesh': 'Amaravati',
    'Arunachal Pradesh': 'Itanagar',
    'Assam': 'Dispur',
    'Bihar': 'Patna',
    'Chhattisgarh': 'Raipur',
    'Goa': 'Panaji',
    'Gujarat': 'Gandhinagar',
    'Haryana': 'Chandigarh',
    'Himachal Pradesh': 'Shimla',
    'Jharkhand': 'Ranchi',
    'Kerala': 'Thiruvananthapuram',
    'Madhya Pradesh': 'Bhopal',
    'Manipur': 'Imphal',
    'Meghalaya': 'Shillong',
    'Mizoram': 'Aizawl',
    'Nagaland': 'Kohima',
    'Odisha': 'Bhubaneswar',
    'Punjab': 'Chandigarh',
    'Sikkim': 'Gangtok',
    'Telangana': 'Hyderabad',
    'Tripura': 'Agartala',
    'Uttarakhand': 'Dehradun',
    'West Bengal': 'Kolkata',
    'Jammu and Kashmir': 'Srinagar',
    'Ladakh': 'Leh',
    'Chandigarh': 'Chandigarh',
    'Puducherry': 'Puducherry',
    'Andaman and Nicobar Islands': 'Port Blair',
    'Dadra and Nagar Haveli and Daman and Diu': 'Daman',
    'Lakshadweep': 'Kavaratti'
}

# Custom clustering for large states
def assign_cluster(state, office_name):
    name = office_name.lower()
    if state == 'Delhi':
        return 'Delhi'
    elif state == 'Uttar Pradesh':
        return 'Noida' if any(k in name for k in ['noida', 'ghaziabad']) else 'Lucknow'
    elif state == 'Maharashtra':
        return 'Pune' if any(k in name for k in ['pune', 'chinchwad']) else 'Mumbai'
    elif state == 'Karnataka':
        return 'Mysuru' if 'mysore' in name or 'mysuru' in name else 'Bengaluru'
    elif state == 'Tamil Nadu':
        return 'Coimbatore' if 'coimbatore' in name else 'Chennai'
    elif state == 'Rajasthan':
        return 'Jodhpur' if 'jodhpur' in name else 'Jaipur'
    else:
        return STATE_CAPITAL_CLUSTERS.get(state, state)

# RTO data processor
def process_rto_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        required_cols = {'state_name', 'office_name', 'registrations'}
        if not required_cols.issubset(df.columns):
            st.error("File must contain: state_name, office_name, registrations")
            return None

        df['Cluster_City'] = df.apply(lambda row: assign_cluster(row['state_name'], row['office_name']), axis=1)
        city_demand = df.groupby('Cluster_City')['registrations'].sum().reset_index()

        # Demand per 1000 (‰)
        city_demand['RTO_Score'] = (city_demand['registrations'] / city_demand['registrations'].sum()) * 1000
        city_demand = city_demand.sort_values('RTO_Score', ascending=False)

        return city_demand[['Cluster_City', 'RTO_Score']].rename(columns={'Cluster_City': 'City'})

    except Exception as e:
        st.error(f"Error processing RTO data: {str(e)}")
        return None

# Main Streamlit App
def main():
    st.title("🚗 RTO-Based Vehicle Demand Clustering")
    st.markdown("""
    Clusters RTO registrations under major cities based on their state and office name.  
    Large states like UP, Maharashtra, TN, etc. are split into two clusters.  
    **Demand score is shown per 1000 registrations (‰)** for better readability.  
    Includes an interactive chart and downloadable table.
    """)

    with st.expander("📁 STEP 1: Upload RTO Data", expanded=True):
        uploaded_file = st.file_uploader("Upload RTO data (CSV or Excel)", type=['csv', 'xlsx'],
                                         help="Must include 'state_name', 'office_name', and 'registrations' columns")

        if uploaded_file:
            rto_data = process_rto_data(uploaded_file)

            if rto_data is not None:
                st.success(f"✅ Clustered and processed {len(rto_data)} cities")
                st.dataframe(rto_data)

                # 📊 Interactive Chart
                with st.expander("📊 View Demand Chart", expanded=True):
                    st.header("📊 Interactive Demand Chart (per 1000 registrations)")

                    fig = px.bar(
                        rto_data,
                        x='City',
                        y='RTO_Score',
                        title="Vehicle Demand by City Cluster",
                        labels={'RTO_Score': 'Demand Score (‰ per 1000)'},
                        hover_data={'City': True, 'RTO_Score': ':.2f'},
                        height=600
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

                # 📋 Data Table + Download
                with st.expander("📋 Detailed Table"):
