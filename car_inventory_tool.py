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
    Clusters regional variations (
