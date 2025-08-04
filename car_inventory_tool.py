import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime

# Initialize page layout
st.set_page_config(layout="wide")

# Function to cluster city regions
def cluster_cities(city_name):
    city_name = str(city_name).lower()
    patterns = [
        r'(.*?)\s*(north|south|east|west|central|urban|rural|greater|metro|municipal).*',
        r'(.*?)\s*(district|division|zone).*',
        r'(.*?)\s*(city|town).*'
    ]
    for pattern in patterns:
        match = re.match(pattern, city_name)
        if match:
            return match.group(1).strip().title()
    return city_name.title()

# Function to process RTO data with city clustering
def process_rto_data(uploaded_file, top_n=30):
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Check required columns
        if 'office_name' not in df.columns or 'registrations' not in df.columns:
            st.error("Uploaded file must contain 'office_name' and 'registrations' columns")
            return None

        # Cluster city regions
        df['City_Cluster'] = df['office_name'].apply(cluster_cities)

        # Aggregate by clustered city
        city_demand = df.groupby('City_Cluster')['registrations'].sum().reset_index()

        # Calculate demand score and get top N cities
        city_demand['RTO_Score'] = (city_demand['registrations'] / city_demand['registrations'].sum()) * 100
        city_demand = city_demand.sort_values('RTO_Score', ascending=False).head(top_n)

        return city_demand[['City_Cluster', 'RTO_Score']].rename(columns={'City_Cluster': 'City'})
    except Exception as e:
        st.error(f"Error processing RTO data: {str(e)}")
        return None

# Main Streamlit app
def main():
    st.title("🚗 Real-time City-wise Vehicle Demand Analysis")
    st.markdown("""
    **Analyzes demand patterns across India's top 30 cities based on RTO registration data.**  
    Clusters regional variations (e.g., Delhi North/South) into single city entries.
    """)

    # STEP 1: Upload data
    with st.expander("📁 STEP 1: Upload RTO Data", expanded=True):
        uploaded_file = st.file_uploader("Upload vehicle registration data (CSV/Excel)",
                                         type=['csv', 'xlsx'],
                                         help="File should contain vehicle registration data")

        if uploaded_file is not None:
            rto_data = process_rto_data(uploaded_file, top_n=30)

            if rto_data is not None:
                st.success(f"✅ Processed RTO data for {len(rto_data)} clustered cities")
                st.dataframe(rto_data)

                # STEP 2: Show visualization (same layout preserved)
                with st.expander("🔍 STEP 2: View Demand Analysis", expanded=True):
                    st.header("📊 Demand Analysis Results - Top 30 Cities")

                    fig, ax = plt.subplots(figsize=(14, 10))
                    cities = rto_data['City']
                    positions = np.arange(len(cities))
                    bar_width = 0.6

                    bars = ax.bar(positions, rto_data['RTO_Score'], bar_width, label='RTO Demand', color='#1f77b4')
                    ax.set_xticks(positions)
                    ax.set_xticklabels(cities, rotation=45, ha='right')
                    ax.set_ylabel("Demand Score")
                    ax.set_title("Vehicle Demand Based on RTO Registrations")
                    ax.legend()

                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.1f}', ha='center', va='bottom', fontsize=8)

                    plt.tight_layout()
                    st.pyplot(fig)

                    # Show data table
                    with st.expander("📋 View Detailed Data"):
                        st.dataframe(rto_data.style
                                     .background_gradient(cmap='Blues', subset=['RTO_Score']),
                                     use_container_width=True)

                    # Download CSV
                    csv = rto_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download RTO Demand Data",
                        data=csv,
                        file_name=f"rto_demand_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime='text/csv'
                    )

if __name__ == "__main__":
    main()
