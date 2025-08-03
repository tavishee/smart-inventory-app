import streamlit as st
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import plotly.express as px

# ------------------ Configuration ------------------
st.set_page_config(page_title="Vehicle Demand Analyzer", layout="wide")

# ------------------ Data Loading Function ------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_rto_data():
    url = "https://ckandev.indiadataportal.com/datastore/dump/cc32d3e2-7ea3-4b6b-94ab-85e57f6a0a3a?format=csv"
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        
        # Verify required columns exist
        required_cols = {'State Name', 'RTO Name', 'Registrations', 'type', 'Categorized', 'Date'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            st.error(f"Missing columns in dataset: {', '.join(missing_cols)}")
            return None
        
        # Data cleaning
        df['Registrations'] = pd.to_numeric(df['Registrations'], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Registrations', 'State Name'])
        
        # State-level aggregation
        state_df = df.groupby('State Name').agg(
            total_registrations=('Registrations', 'sum'),
            unique_rto_count=('RTO Name', 'nunique'),
            vehicle_types=('type', 'nunique'),
            transport_ratio=('Categorized', lambda x: (x == 'Transport').mean()),
            latest_date=('Date', 'max')
        ).reset_index()
        
        # Calculate normalized scores
        state_df['norm_registration'] = (state_df['total_registrations'] - state_df['total_registrations'].min()) / \
                                      (state_df['total_registrations'].max() - state_df['total_registrations'].min())
        
        return state_df.rename(columns={'State Name': 'state'})
    
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return None

# ------------------ Google Trends Function ------------------
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_trends_score(location):
    pytrends = TrendReq(hl='en-US', tz=330)
    
    queries = [
        f"used cars in {location}",
        f"buy car {location}",
        f"second hand car {location}",
        f"car dealership {location}"
    ]
    
    try:
        pytrends.build_payload(queries, geo='IN', timeframe='today 3-m')
        trends_df = pytrends.interest_over_time()
        if not trends_df.empty:
            return trends_df[queries].mean().mean()
    except:
        pass
    
    return 10.0  # Default score if no trends found

# ------------------ Main App ------------------
def main():
    st.title("🚗 India Vehicle Demand Analysis")
    st.markdown("Analyzing vehicle registration trends across Indian states")
    
    # Load data
    with st.spinner("Loading vehicle registration data..."):
        rto_data = load_rto_data()
    
    if rto_data is None:
        st.error("Failed to load vehicle registration data. Please try again later.")
        return
    
    # Calculate trends
    with st.spinner("Fetching Google Trends data (this may take a few minutes)..."):
        rto_data['trend_score'] = rto_data['state'].apply(get_trends_score)
        rto_data['norm_trend'] = (rto_data['trend_score'] - rto_data['trend_score'].min()) / \
                                (rto_data['trend_score'].max() - rto_data['trend_score'].min())
    
    # User controls
    st.sidebar.header("Analysis Parameters")
    trend_weight = st.sidebar.slider("Google Trends Weight", 0.0, 1.0, 0.3)
    min_rto = st.sidebar.slider("Minimum RTO Count", 1, int(rto_data['unique_rto_count'].max()), 1)
    
    # Calculate combined score
    rto_data['combined_score'] = (1 - trend_weight) * rto_data['norm_registration'] + \
                                trend_weight * rto_data['norm_trend']
    
    # Filter data
    filtered_data = rto_data[rto_data['unique_rto_count'] >= min_rto].sort_values('combined_score', ascending=False)
    
    # ------------------ Visualizations ------------------
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top States by Demand Score")
        fig1 = px.bar(filtered_data.head(10), 
                     x='state', y='combined_score',
                     color='norm_registration',
                     labels={'combined_score': 'Demand Score', 'state': 'State'},
                     hover_data=['total_registrations', 'unique_rto_count'])
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Transport vs Non-Transport Ratio")
        fig2 = px.pie(filtered_data, 
                     names='state', values='transport_ratio',
                     title='Transport Vehicle Percentage by State')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Map visualization
    st.subheader("Geographical Distribution")
    fig3 = px.choropleth(filtered_data,
                        locations='state',
                        locationmode='country names',
                        color='combined_score',
                        scope='asia',
                        hover_name='state',
                        hover_data=['total_registrations', 'trend_score'],
                        color_continuous_scale='Viridis',
                        title='Vehicle Demand Heatmap')
    fig3.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Raw data
    st.subheader("Detailed Data")
    st.dataframe(filtered_data.style.background_gradient(cmap='Blues', subset=['combined_score']))

if __name__ == "__main__":
    main()

