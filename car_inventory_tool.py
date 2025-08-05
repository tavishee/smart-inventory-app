import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import re
from datetime import datetime

# -------------------------
# City Clustering by State Capitals
# -------------------------

# 33 state clusters (Delhi is standalone, 5 states have 2 clusters)
city_cluster_map = {
    'Uttar Pradesh': ['Lucknow', 'Noida'],
    'Maharashtra': ['Mumbai', 'Pune'],
    'Karnataka': ['Bengaluru', 'Mysuru'],
    'Tamil Nadu': ['Chennai', 'Coimbatore'],
    'Rajasthan': ['Jaipur', 'Jodhpur'],
    'Delhi': ['Delhi'],
    'Andhra Pradesh': ['Amaravati'],
    'Arunachal Pradesh': ['Itanagar'],
    'Assam': ['Dispur'],
    'Bihar': ['Patna'],
    'Chhattisgarh': ['Raipur'],
    'Goa': ['Panaji'],
    'Gujarat': ['Gandhinagar'],
    'Haryana': ['Chandigarh'],
    'Himachal Pradesh': ['Shimla'],
    'Jharkhand': ['Ranchi'],
    'Kerala': ['Thiruvananthapuram'],
    'Madhya Pradesh': ['Bhopal'],
    'Manipur': ['Imphal'],
    'Meghalaya': ['Shillong'],
    'Mizoram': ['Aizawl'],
    'Nagaland': ['Kohima'],
    'Odisha': ['Bhubaneswar'],
    'Punjab': ['Chandigarh'],
    'Sikkim': ['Gangtok'],
    'Telangana': ['Hyderabad'],
    'Tripura': ['Agartala'],
    'Uttarakhand': ['Dehradun'],
    'West Bengal': ['Kolkata'],
}

# Flatten the map to assign each RTO to its cluster
rto_cluster_map = {}
for state, clusters in city_cluster_map.items():
    for city in clusters:
        rto_cluster_map[city.lower()] = city

def assign_cluster_by_state_office_name(office_name):
    name = str(office_name).lower()
    for cluster_key in rto_cluster_map:
        if cluster_key in name:
            return rto_cluster_map[cluster_key]
    return "Other"

# -------------------------
# Process RTO Data
# -------------------------
def process_rto_data(uploaded_file, top_n=34):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if 'office_name' not in df.columns or 'registrations' not in df.columns or 'class_type' not in df.columns:
            st.error("Uploaded file must contain 'office_name', 'registrations', and 'class_type' columns")
            return None

        # Assign clusters
        df['City_Cluster'] = df['office_name'].apply(assign_cluster_by_state_office_name)

        # Total registrations per cluster
        total_regs = df.groupby('City_Cluster')['registrations'].sum().reset_index(name='Total_Registrations')

        # Vehicle class weighting
        weights = {
            'SUV': 1.2,
            'Sedan': 1.0,
            'Hatchback': 0.9,
            'Two-Wheeler': 0.5,
            'Three-Wheeler': 0.4,
            'Tractor': 0.3,
            'LCV': 0.8,
            'MCV': 0.6,
            'HCV': 0.5
        }

        df['Class_Weight'] = df['class_type'].map(weights).fillna(0.5)
        df['Weighted_Class_Score'] = df['registrations'] * df['Class_Weight']

        class_scores = df.groupby('City_Cluster')['Weighted_Class_Score'].sum().reset_index(name='Class_Weighted')

        # Merge
        merged = pd.merge(total_regs, class_scores, on='City_Cluster')

        # Normalize scores
        merged['Volume_Score'] = (merged['Total_Registrations'] / merged['Total_Registrations'].sum()) * 1000
        merged['Class_Score'] = (merged['Class_Weighted'] / merged['Class_Weighted'].sum()) * 1000

        # Final composite score
        merged['Buying_Strength_Score'] = (
            0.7 * merged['Volume_Score'] +
            0.3 * merged['Class_Score']
        )

        result = merged.sort_values('Buying_Strength_Score', ascending=False).head(top_n)
        return result

    except Exception as e:
        st.error(f"Error processing RTO data: {str(e)}")
        return None

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config(layout="wide")
    st.title("🚗 Used Car Market - City-wise Buying Strength Analysis")
    st.markdown("""
    This tool ranks Indian cities by **used car buying strength**, based on:
    - Total vehicle registrations
    - Weighted mix of vehicle classes (SUVs, Sedans, Bikes, etc.)
    
    The final score reflects **overall demand potential**, not just for one fuel or model type.
    """)

    with st.expander("📁 Upload RTO Data", expanded=True):
        uploaded_file = st.file_uploader("Upload vehicle registration data (CSV/Excel)",
                                         type=['csv', 'xlsx'],
                                         help="Must include columns: office_name, registrations, class_type")

        if uploaded_file:
            df = process_rto_data(uploaded_file, top_n=34)
            if df is not None:
                st.success("✅ Processed successfully.")
                st.dataframe(df[['City_Cluster', 'Volume_Score', 'Class_Score', 'Buying_Strength_Score']])

                # Plotly interactive chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=df['City_Cluster'],
                    x=df['Volume_Score'],
                    name='Volume Score',
                    orientation='h',
                    marker_color='steelblue'
                ))
                fig.add_trace(go.Bar(
                    y=df['City_Cluster'],
                    x=df['Class_Score'],
                    name='Class Score',
                    orientation='h',
                    marker_color='seagreen'
                ))
                fig.update_layout(
                    barmode='stack',
                    title='📊 Buying Strength Score by City Cluster',
                    xaxis_title='Composite Demand Score (0–1000 scale)',
                    yaxis_title='City Cluster',
                    height=800,
                    legend=dict(orientation="h", y=-0.2)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Download
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=df.to_csv(index=False),
                    file_name=f"buying_strength_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv'
                )

if __name__ == "__main__":
    main()

