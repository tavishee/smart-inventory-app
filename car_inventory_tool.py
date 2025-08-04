import streamlit as st
import pandas as pd
import numpy as np
import pytrends
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import re
from datetime import datetime

# Initialize pytrends
pytrends = TrendReq(hl='en-US', tz=360)

# Function to cluster city regions
def cluster_cities(city_name):
    # Remove regional suffixes and standardize city names
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
        
        # Check required columns - we'll use 'office_name' as city and sum registrations
        if 'office_name' not in df.columns or 'registrations' not in df.columns:
            st.error("Uploaded file must contain 'office_name' and 'registrations' columns")
            return None
        
        # Cluster city regions
        df['City_Cluster'] = df['office_name'].apply(cluster_cities)
        
        # Aggregate by clustered city
        city_demand = df.groupby('City_Cluster')['registrations'].sum().reset_index()
        
        # Calculate demand score and get top 30 cities
        city_demand['RTO_Score'] = (city_demand['registrations'] / city_demand['registrations'].sum()) * 100
        city_demand = city_demand.sort_values('RTO_Score', ascending=False).head(top_n)
        
        return city_demand[['City_Cluster', 'RTO_Score']].rename(columns={'City_Cluster': 'City'})
    except Exception as e:
        st.error(f"Error processing RTO data: {str(e)}")
        return None

# Enhanced Google Trends data fetcher with retries
def get_google_trends_data(keywords, cities, timeframe='today 1-m', retries=3):
    trends_data = pd.DataFrame()
    
    for city in cities:
        for keyword in keywords:
            attempts = 0
            success = False
            trends_score = 1  # Default minimum score
            
            while not success and attempts < retries:
                try:
                    # Try with city name + keyword
                    query = f"{keyword} {city.split()[0]}"  # Use first word of city name
                    pytrends.build_payload([query], geo='IN', timeframe=timeframe)
                    city_data = pytrends.interest_by_region(resolution='CITY', inc_low_vol=True)
                    
                    if city_data.empty:
                        # Try with just the keyword
                        pytrends.build_payload([keyword], geo='IN', timeframe=timeframe)
                        city_data = pytrends.interest_by_region(resolution='CITY', inc_low_vol=True)
                    
                    if not city_data.empty:
                        # Get maximum interest value for this city across possible matches
                        city_pattern = re.compile(rf"{re.escape(city.split()[0])}", re.IGNORECASE)
                        matching_cities = [c for c in city_data.index if city_pattern.search(c)]
                        
                        if matching_cities:
                            trends_score = max(city_data.loc[matching_cities].max())
                        else:
                            trends_score = 1
                    
                    success = True
                    
                except Exception as e:
                    attempts += 1
                    if attempts == retries:
                        st.warning(f"Couldn't get trends data for {city} - {keyword}. Using default score.")
                        trends_score = 1
            
            trends_data = pd.concat([
                trends_data,
                pd.DataFrame({'City': [city], 'Keyword': [keyword], 'Trends_Score': [trends_score]})
            ])
    
    # Aggregate by city (average across keywords)
    city_trends = trends_data.groupby('City')['Trends_Score'].mean().reset_index()
    city_trends['Trends_Score'] = (city_trends['Trends_Score'] / city_trends['Trends_Score'].max()) * 100
    
    return city_trends

# Main Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title("🚗 Real-time City-wise Vehicle Demand Analysis")
    st.markdown("""
    **Combines RTO registration data with Google Trends to analyze demand patterns across India's top 30 cities.**
    Clusters regional variations (e.g., Delhi North/South) into single city entries.
    """)
    
    # File upload section
    with st.expander("📁 STEP 1: Upload RTO Data", expanded=True):
        uploaded_file = st.file_uploader("Upload vehicle registration data (CSV/Excel)", 
                                       type=['csv', 'xlsx'],
                                       help="File should contain vehicle registration data")
    
    if uploaded_file is not None:
        rto_data = process_rto_data(uploaded_file, top_n=30)
        
        if rto_data is not None:
            st.success(f"✅ Processed RTO data for {len(rto_data)} clustered cities")
            st.dataframe(rto_data)
            
            # Keyword input section
            with st.expander("🔍 STEP 2: Configure Demand Analysis", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    keywords = st.text_area("Enter vehicle-related keywords (one per line)", 
                                          "electric vehicle\nSUV\nsedan\ncompact car\ntwo-wheeler",
                                          help="Use generic terms that people might search for")
                with col2:
                    timeframe = st.selectbox("Analysis timeframe", 
                                           ["today 1-m", "today 3-m", "today 12-m"],
                                           index=1)
                    weight_rto = st.slider("RTO Data Weight", 0.1, 0.9, 0.5)
                    weight_trends = 1 - weight_rto
            
            keyword_list = [k.strip() for k in keywords.split('\n') if k.strip()]
            
            if st.button("🚀 Analyze Demand", type="primary"):
                with st.spinner("Fetching Google Trends data and analyzing..."):
                    # Get trends data
                    trends_data = get_google_trends_data(keyword_list, rto_data['City'].tolist(), timeframe)
                    
                    # Merge with RTO data
                    merged_data = pd.merge(rto_data, trends_data, on='City', how='left')
                    
                    # Calculate combined score
                    merged_data['Combined_Score'] = (
                        weight_rto * merged_data['RTO_Score'] + 
                        weight_trends * merged_data['Trends_Score']
                    )
                    
                    # Sort by combined score
                    merged_data = merged_data.sort_values('Combined_Score', ascending=False)
                    
                    # Visualization
                    st.header("📊 Demand Analysis Results - Top 30 Cities")
                    
                    # Create a single figure with all cities
                    fig, ax = plt.subplots(figsize=(14, 10))
                    
                    # Set positions for bars
                    cities = merged_data['City']
                    positions = np.arange(len(cities))
                    bar_width = 0.35
                    
                    # Plot bars
                    rto_bars = ax.bar(positions - bar_width/2, merged_data['RTO_Score'], 
                                     bar_width, label='RTO Demand', color='#1f77b4')
                    combined_bars = ax.bar(positions + bar_width/2, merged_data['Combined_Score'], 
                                          bar_width, label='Combined Demand', color='#ff7f0e')
                    
                    # Customize plot
                    ax.set_xticks(positions)
                    ax.set_xticklabels(cities, rotation=45, ha='right')
                    ax.set_ylabel("Demand Score")
                    ax.set_title("Vehicle Demand Comparison Across Cities")
                    ax.legend()
                    
                    # Add value labels on bars
                    for bars in [rto_bars, combined_bars]:
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{height:.1f}',
                                    ha='center', va='bottom', fontsize=8)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show data table
                    with st.expander("📋 View Detailed Data"):
                        st.dataframe(merged_data.style
                                   .background_gradient(cmap='Blues', subset=['RTO_Score'])
                                   .background_gradient(cmap='Oranges', subset=['Trends_Score'])
                                   .background_gradient(cmap='Greens', subset=['Combined_Score']),
                                   use_container_width=True)
                    
                    # Download button
                    csv = merged_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Analysis Results",
                        data=csv,
                        file_name=f"city_demand_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime='text/csv'
                    )

if __name__ == "__main__":
    main()
