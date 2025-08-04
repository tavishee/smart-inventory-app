import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
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

# Special dual-cluster logic for large states
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
        return STATE_CAPITAL_CLUSTERS.get(state, state)  # fallback to state name

# Process uploaded file and cluster data
def process_rto_data(uploaded_file, top_n=30):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if 'office_name' not in df.columns or 'registrations' not in df.columns or 'state_name' not in df.columns:
            st.error("File must contain 'state_name', 'office_name', and 'registrations'")
            return None

        df['Cluster_City'] = df.apply(lambda row: assign_cluster(row['state_name'], row['office_name']), axis=1)

        city_demand = df.groupby('Cluster_City')['registrations'].sum().reset_index()
        city_demand['RTO_Score'] = (city_demand['registrations'] / city_demand['registrations'].sum()) * 100
        city_demand = city_demand.sort_values('RTO_Score', ascending=False)

        return city_demand[['Cluster_City', 'RTO_Score']].rename(columns={'Cluster_City': 'City'})

    except Exception as e:
        st.error(f"Error processing RTO data: {str(e)}")
        return None

# Main app
def main():
    st.title("🚗 City-wise Vehicle Demand Clustering via RTO Data")
    st.markdown("""
    **Clusters RTO registrations under major cities based on their state and office name.**  
    Large states like UP, Maharashtra, TN, etc. are split into two clusters based on demand centers.
    """)

    with st.expander("📁 STEP 1: Upload RTO Data", expanded=True):
        uploaded_file = st.file_uploader("Upload vehicle registration data (CSV/Excel)",
                                         type=['csv', 'xlsx'],
                                         help="File must include 'state_name', 'office_name', and 'registrations'")

        if uploaded_file:
            rto_data = process_rto_data(uploaded_file, top_n=30)

            if rto_data is not None:
                st.success(f"✅ Clustered and processed data for {len(rto_data)} top cities")
                st.dataframe(rto_data)

                with st.expander("📊 STEP 2: View Demand Analysis", expanded=True):
                    st.header("📊 Demand by City Cluster (Top 30)")

                    fig, ax = plt.subplots(figsize=(14, 10))
                    cities = rto_data['City']
                    positions = np.arange(len(cities))
                    bar_width = 0.6

                    bars = ax.bar(positions, rto_data['RTO_Score'], bar_width,
                                  label='RTO Demand', color='#1f77b4')
                    ax.set_xticks(positions)
                    ax.set_xticklabels(cities, rotation=45, ha='right')
                    ax.set_ylabel("Demand Score (%)")
                    ax.set_title("RTO Vehicle Demand Clustered by City")
                    ax.legend()

                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.1f}', ha='center', va='bottom', fontsize=8)

                    plt.tight_layout()
                    st.pyplot(fig)

                    with st.expander("📋 View Detailed Data"):
                        st.dataframe(rto_data.style
                                     .background_gradient(cmap='Blues', subset=['RTO_Score']),
                                     use_container_width=True)

                    csv = rto_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Clustered Demand Data",
                        data=csv,
                        file_name=f"rto_clustered_demand_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime='text/csv'
                    )

if __name__ == "__main__":
    main()

