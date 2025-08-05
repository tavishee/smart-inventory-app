
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from datetime import datetime

# -------------------------
# Custom State-Based Clustering
# -------------------------
# State/UT area in km²
state_area_km2 = {
    'RJ': 342239, 'MP': 308252, 'MH': 307713, 'UP': 240928, 'GJ': 196024,
    'KA': 191791, 'AP': 162975, 'OD': 155707, 'CG': 135192, 'TN': 130058,
    'TG': 112077, 'BR': 94163, 'WB': 88752, 'AR': 83743, 'JH': 79716,
    'AS': 78438, 'HP': 55673, 'UK': 53483, 'PB': 50362, 'HR': 44212,
    'KL': 38863, 'ML': 22429, 'MN': 22327, 'MZ': 21081, 'NL': 16579,
    'TR': 10491, 'SK': 7096, 'GA': 3702, 'AN': 8249, 'JK': 42241,
    'LA': 59146, 'DL': 1484, 'DD': 603, 'PY': 479, 'CH': 114, 'LD': 32
}

def get_cluster_area(cluster):
    """Return area of cluster in km². Split clusters get half area."""
    if '_' in cluster:  # e.g., MH_A, UP_B
        code = cluster.split('_')[0]
        return state_area_km2.get(code, 0) / 2
    return state_area_km2.get(cluster, 0)


# Large states split into two clusters
split_states = {'MH', 'UP', 'RJ', 'TN', 'KA'}

def extract_office_prefix(office_code):
    match = re.match(r'([A-Z]{2})\d+', str(office_code).strip().upper())
    return match.group(1) if match else None

def assign_state_cluster(row):
    code = row['state_code']
    if code in split_states:
        office = str(row['office_code']).strip().upper()
        num_part = re.sub(r'[A-Z]', '', office)
        if num_part.isdigit() and int(num_part) >= 50:
            return f"{code}_B"
        else:
            return f"{code}_A"
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

        required_cols = {'office_name', 'office_code', 'registrations', 'class_type'}
        if not required_cols.issubset(set(df.columns)):
            st.error("Uploaded file must contain 'office_name', 'office_code', 'registrations', and 'class_type' columns")
            return None

        # Clean and filter class_type
        df['class_type'] = df['class_type'].astype(str).str.strip().str.lower()
        allowed_classes = {'motor car', 'luxury cab', 'maxi cab', 'm-cycle/scooter'}
        df = df[df['class_type'].isin(allowed_classes)]

        # Extract state prefix from office_code
        df['state_code'] = df['office_code'].apply(extract_office_prefix)

        # Assign clusters
        df['City_Cluster'] = df.apply(assign_state_cluster, axis=1)
        df = df[df['City_Cluster'].notna()]

        # Check if data is empty
        if df.empty:
            st.warning("Filtered dataset is empty. Please check class_type or office_code values.")
            return None

        # Group and score
        cluster_scores = df.groupby('City_Cluster')['registrations'].sum().reset_index(name='Total_Registrations')
        cluster_scores['Volume_Score'] = (cluster_scores['Total_Registrations'] / cluster_scores['Total_Registrations'].sum()) * 1000
        cluster_scores['Buying_Strength_Score'] = cluster_scores['Volume_Score']

        result = cluster_scores.sort_values('Buying_Strength_Score', ascending=False).head(top_n)
        return result

    except Exception as e:
        st.error(f"Error processing RTO data: {str(e)}")
        return None

        # Clean and filter class_type
        df['class_type'] = df['class_type'].astype(str).str.strip().str.lower()
        allowed_classes = {'motor car', 'luxury cab', 'maxi cab', 'M-Cycle/Scooter'}
        df = df[df['class_type'].isin(allowed_classes)]

        # Extract state code and assign clusters
        df['office_code'] = df['office_name'].apply(office_code)
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

        # Add cluster area (in km²)
        result['Cluster_Area_km2'] = result['City_Cluster'].apply(get_cluster_area)

        # Add Demand Density (registrations per 1000 km²)
        result['Demand_Density_per_1000_km2'] = (result['Total_Registrations'] / result['Cluster_Area_km2']) * 1000


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
    - Total vehicle registrations (filtered for Motor Car, Luxury Cab, Maxi Cab, M Cycle, Scooter)
    
    The final score reflects **overall demand potential**, not just for one fuel or model type.
    """)

    with st.expander("📁 Upload RTO Data", expanded=True):
        uploaded_file = st.file_uploader("Upload vehicle registration data (CSV/Excel)",
                                         type=['csv', 'xlsx'],
                                         help="Must include columns: office_name, registrations, class_type")

        if uploaded_file:
            result = process_rto_data(uploaded_file, top_n=34)
            if result is not None:
                st.success("✅ Processed successfully.")
                st.dataframe(result[['City_Cluster', 'Total_Registrations', 'Volume_Score', 'Buying_Strength_Score', 'Demand_Density_per_1000_km2']])


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







