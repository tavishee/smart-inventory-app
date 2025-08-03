import streamlit as st
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import plotly.express as px

# ------------------ Fixed Data Loading Function ------------------
@st.cache_data
def load_rto():
    url = "https://ckandev.indiadataportal.com/datastore/dump/cc32d3e2-7ea3-4b6b-94ab-85e57f6a0a3a?format=csv"
    try:
        # Load and clean data
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        
        # Debug: Show available columns
        st.write("Available columns:", df.columns.tolist())
        
        # Check if required columns exist
        if 'Registrations' not in df.columns:
            if 'Registration' in df.columns:  # Check for singular form
                df['Registrations'] = df['Registration']
            else:
                st.error("Registrations column not found in dataset")
                return pd.DataFrame(columns=["city", "rto_total"])
        
        # Handle state name column (different versions in different datasets)
        state_col = None
        for col in ['State Name', 'State_Name', 'Name', 'state']:
            if col in df.columns:
                state_col = col
                break
                
        if not state_col:
            st.error("No state name column found")
            return pd.DataFrame(columns=["city", "rto_total"])
        
        # Convert registrations to numeric
        df['Registrations'] = pd.to_numeric(df['Registrations'], errors='coerce')
        df = df[df['Registrations'].notna()]
        
        # Group by state
        df_grouped = df.groupby(state_col)['Registrations'].sum().reset_index(name="rto_total")
        df_grouped["city"] = df_grouped[state_col]
        
        return df_grouped[["city", "rto_total"]]
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(columns=["city", "rto_total"])

# ------------------ Google Trends Function ------------------
@st.cache_data
def fetch_trends(city):
    try:
        pytrends = TrendReq(hl='en-US', tz=330)
        queries = [f"used cars {city}", f"buy car {city}", f"second hand car {city}"]
        pytrends.build_payload(queries, geo='IN', timeframe='today 3-m')
        df = pytrends.interest_over_time()
        return df[queries].mean().mean() if not df.empty else 10.0
    except:
        return 10.0  # Default value if API fails

# ------------------ Main App ------------------
def main():
    st.title("Vehicle Demand Analysis")
    
    # Load data
    rto_df = load_rto()
    
    if rto_df.empty:
        st.warning("No data loaded. Check the data source.")
        return
    
    # Normalize RTO data
    rto_df["score_rto"] = (rto_df["rto_total"] - rto_df["rto_total"].min()) / \
                         (rto_df["rto_total"].max() - rto_df["rto_total"].min())
    
    # Get trends
    st.info("Fetching Google Trends...")
    rto_df["trend_score"] = rto_df["city"].apply(fetch_trends)
    rto_df["score_trend_norm"] = (rto_df["trend_score"] - rto_df["trend_score"].min()) / \
                               (rto_df["trend_score"].max() - rto_df["trend_score"].min())
    
    # Combined score
    alpha = st.sidebar.slider("Google Trends Weight", 0.0, 1.0, 0.5)
    rto_df["score_combined"] = (1 - alpha) * rto_df["score_rto"] + alpha * rto_df["score_trend_norm"]
    
    # Display
    selected_city = st.selectbox("Select Location", sorted(rto_df["city"].unique()))
    
    # Plot
    fig = px.bar(rto_df.sort_values("score_combined", ascending=False).head(20),
                x="city", y="score_combined",
                title="Top Locations by Vehicle Demand Score")
    st.plotly_chart(fig)
    
    # Show raw data
    st.dataframe(rto_df.sort_values("score_combined", ascending=False))

if __name__ == "__main__":
    main()
