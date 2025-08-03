import streamlit as st
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import plotly.express as px

# ------------------ Debugging Version of Data Loading Function ------------------
@st.cache_data
def load_rto():
    url = "https://ckandev.indiadataportal.com/datastore/dump/cc32d3e2-7ea3-4b6b-94ab-85e57f6a0a3a?format=csv"
    try:
        # Load the data with verbose debugging
        st.write("⚠️ DEBUG: Loading data from URL...")
        df = pd.read_csv(url)
        st.write("✅ DEBUG: Data loaded successfully")
        
        # Show raw column names before cleaning
        st.write("🔍 DEBUG: Original columns:", df.columns.tolist())
        
        # Clean column names
        df.columns = df.columns.str.strip()
        st.write("🧹 DEBUG: After stripping whitespace:", df.columns.tolist())
        
        # Find the correct column names with fuzzy matching
        registrations_col = None
        state_col = None
        rto_col = None
        
        # Check for all possible column name variations
        for col in df.columns:
            col_lower = col.lower()
            if 'registr' in col_lower:
                registrations_col = col
            if 'state' in col_lower or 'name' in col_lower:
                state_col = col
            if 'rto' in col_lower:
                rto_col = col
        
        # Debug column detection
        st.write(f"🔎 DEBUG: Detected columns - Registrations: {registrations_col}, State: {state_col}, RTO: {rto_col}")
        
        # Validate we found required columns
        if not registrations_col:
            st.error("❌ ERROR: Could not find Registrations column in: " + str(df.columns.tolist()))
            return pd.DataFrame()
        
        if not state_col:
            st.warning("⚠️ WARNING: Could not find State column, using first column as state")
            state_col = df.columns[0]
        
        # Convert registrations to numeric
        df[registrations_col] = pd.to_numeric(df[registrations_col], errors='coerce')
        df = df[df[registrations_col].notna()]
        
        # Group by state (or whatever column we detected)
        st.write(f"📊 DEBUG: Grouping by {state_col} with column {registrations_col}")
        df_grouped = df.groupby(state_col)[registrations_col].sum().reset_index(name="rto_total")
        df_grouped["city"] = df_grouped[state_col]
        
        st.write("🧮 DEBUG: First 5 rows of processed data:", df_grouped.head())
        return df_grouped[["city", "rto_total"]]
    
    except Exception as e:
        st.error(f"🔥 CRITICAL ERROR: {str(e)}")
        return pd.DataFrame(columns=["city", "rto_total"])

# ------------------ Main App ------------------
def main():
    st.title("🚗 Vehicle Demand Analyzer")
    st.write("This version includes extensive debugging to identify column name issues")
    
    # Load data with debugging
    rto_df = load_rto()
    
    if rto_df.empty:
        st.error("No data was loaded. Please check the debug output above.")
        st.stop()
    
    st.success(f"Successfully loaded data with {len(rto_df)} entries")
    st.dataframe(rto_df.head())

if __name__ == "__main__":
    main()
