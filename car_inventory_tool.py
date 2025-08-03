import streamlit as st
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import plotly.express as px

# ------------------ Load RTO to City Mapping ------------------
rto_city_map = {
    "DELHI-NORTH": "Delhi", "DELHI-SOUTH": "Delhi", "DELHI-EAST": "Delhi",
    "MUMBAI-WEST": "Mumbai", "MUMBAI-EAST": "Mumbai", "THANE": "Mumbai",
    "BENGALURU-EAST": "Bengaluru", "BENGALURU-WEST": "Bengaluru",
    "HYDERABAD-CENTRAL": "Hyderabad", "HYDERABAD-NORTH": "Hyderabad",
    "CHENNAI-SOUTH": "Chennai", "CHENNAI-NORTH": "Chennai",
    "KOLKATA (SALT LAKE)": "Kolkata", "KOLKATA (TOLLYGUNGE)": "Kolkata",
    "PUNE": "Pune", "NAVI MUMBAI": "Mumbai", "GURGAON": "Gurgaon",
    "FARIDABAD": "Delhi NCR", "LUCKNOW": "Lucknow", "AHMEDABAD": "Ahmedabad",
    "JAIPUR": "Jaipur", "CHANDIGARH": "Chandigarh", "PATNA": "Patna",
    "SURAT": "Surat", "BHOPAL": "Bhopal", "NAGPUR": "Nagpur"
    # Extend with more as needed
}

# ------------------ Load RTO Data Automatically from URL ------------------
@st.cache_data
def load_rto():
    url = "https://ckandev.indiadataportal.com/datastore/dump/cc32d3e2-7ea3-4b6b-94ab-85e57f6a0a3a?format=csv"  # Replace with actual public CSV URL
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()  # Remove any leading/trailing whitespace or BOM
    df["city"] = df["RTO Name"].map(rto_city_map)
    df = df[df["city"].notna()]  # drop rows with no mapped city
    df_city = df.groupby("city")["Registrations"].sum().reset_index()
    df_city.rename(columns={"Registrations": "rto_total"}, inplace=True)
    return df_city

rto_df = load_rto()

# Normalize RTO demand
rto_df["score_rto"] = (rto_df["rto_total"] - rto_df["rto_total"].min()) / (
    rto_df["rto_total"].max() - rto_df["rto_total"].min()
)

# ------------------ Google Trends Fetch with Fallback ------------------
def fetch_trends(city):
    pytrends = TrendReq(hl='en-US', tz=330)

    fallback_queries = ["used car India", "second hand car", "buy used car", "used cars near me"]
    city_queries = [
        f"{city} used car", f"used cars in {city}",
        f"second hand car {city}", f"buy used car {city}"
    ]

    try:
        pytrends.build_payload(city_queries, geo='IN', timeframe='now 7-d')
        df = pytrends.interest_over_time()
        valid_cols = [col for col in city_queries if col in df.columns]
        if not valid_cols or df.empty:
            raise ValueError("City-level trend missing")

        score = df[valid_cols].mean().mean()
        if np.isnan(score) or score < 1:
            raise ValueError("Low score")
        return score

    except:
        try:
            pytrends.build_payload(fallback_queries, geo='IN', timeframe='now 7-d')
            df_fb = pytrends.interest_over_time()
            valid_cols_fb = [col for col in fallback_queries if col in df_fb.columns]
            fallback_score = df_fb[valid_cols_fb].mean().mean()
            return fallback_score if not np.isnan(fallback_score) else 10.0
        except:
            return 10.0

# ------------------ Compute Trend Scores ------------------
st.info("Fetching Google Trends... Please wait a few seconds...")
rto_df["trend_score"] = rto_df["city"].apply(fetch_trends)

# Normalize trend scores
min_t, max_t = np.nanmin(rto_df["trend_score"]), np.nanmax(rto_df["trend_score"])
rto_df["score_trend_norm"] = (rto_df["trend_score"] - min_t) / (max_t - min_t)

# ------------------ Combined Score ------------------
alpha = st.sidebar.slider("Weight for Google Trends", 0.0, 1.0, 0.5, step=0.1)
rto_df["score_combined"] = (1 - alpha) * rto_df["score_rto"] + alpha * rto_df["score_trend_norm"]

# ------------------ Visual Output ------------------
st.title("Real-time City-wise Car Demand Map")

selected_city = st.selectbox("Select City", sorted(rto_df["city"].unique()))
row = rto_df[rto_df["city"] == selected_city].iloc[0]

df_plot = pd.DataFrame({
    "score_type": ["RTO Only", "RTO + Google Trends"],
    "score": [row.score_rto, row.score_combined]
})

fig = px.bar(df_plot, x="score_type", y="score",
             title=f"Demand Scores for {selected_city}",
             labels={"score": "Normalized Demand Score"},
             color="score_type",
             text="score")

st.plotly_chart(fig, use_container_width=True)

st.write("## Full City-wise Score Table")
st.dataframe(rto_df.sort_values("score_combined", ascending=False))
