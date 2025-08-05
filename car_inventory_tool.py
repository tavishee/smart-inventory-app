
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from datetime import datetime

# -------------------------
# Custom State-Based Clustering
# -------------------------

# Large states split into two clusters
split_states = {'MH', 'UP', 'RJ', 'TN', 'KA'}

def extract_state_code(office_name):
    match = re.search(r'\b([A-Z]{2})\d{1,2}', str(office_name).upper())
    return match.group(1) if match else None

def assign_state_cluster(row):
    code = row['state_code']
    if code in split_states:
        return f"{code}_A" if hash(row['office_name']) % 2 == 0 else f"{code}_B"
    return code

# -------------------------
# Process RTO Data
# -------------------------
def process_rto_data(uploaded_file, top_n=34):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        required_cols = {'office_name', 'registrations', 'class_type'}
        if not required_cols.issubset(set(df.columns)):
            st.error("Uploaded file must contain 'office_name', 'registrations', and 'class_type' columns")
            return None

        # Clean and filter class_type
        df['class_type'] = df['class_type'].astype(str).str.strip().str.lower()
        allowed_classes = {'motor car', 'luxury cab', 'maxi cab', 'm cycle', 'scooter'}
        df = df[df['class_type'].isin(allowed_classes)]

        # Extract state code and assign clusters
        df['state_code'] = df['office_name'].apply(extract_state_code)
        df['City_Cluster'] = df.apply(assign_state_cluster, axis=1)
        df = df[df['City_Cluster'].notna()]

        # Check if data is empty
        if df.empty:
            st.warning("Filtered dataset is empty. Please check class_type or office_name values.")
            return None

        # Group by cluster and calculate scores
        cluster_scores = df.groupby('City_Cluster')['registrations'].sum().reset_index(name='Total_Registrations')
        cluster_scores['Volume_Score'] = (cluster_scores['Total_Registrations'] / cluster_scores['Total_Registrations'].sum()) * 1000
        cluster_scores['Buying_Strength_Score'] = cluster_scores['Volume_Score']

        result = cluster_scores.sort_values('Buying_Strength_Score', ascending=False).head(top_n)
        return result

    except Exception as e:
        st.error(f"Error processing RTO data: {str(e)}")
        return None

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config(layout="wide")
    st.title("🚗 Used Car Market - State-wise Buying Strength Analysis")
    st.markdown("""
    This tool ranks Indian states by **used car buying strength**, based on:
    - Total vehicle registrations (filtered for Motor Car, Luxury Cab, Maxi Cab, M-Cycle/Scooter)
    
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
                st.dataframe(df[['City_Cluster', 'Volume_Score', 'Buying_Strength_Score']])

                # Plotly interactive chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=df['City_Cluster'],
                    x=df['Volume_Score'],
                    name='Volume Score',
                    orientation='h',
                    marker_color='steelblue'
                ))
                fig.update_layout(
                    title='📊 Buying Strength Score by State Cluster',
                    xaxis_title='Composite Demand Score (0–1000 scale)',
                    yaxis_title='State Cluster',
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
