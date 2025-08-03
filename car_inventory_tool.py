import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from api_utils import get_distance_km, get_trend_score_all, get_fuel_price, ALL_TREND_KEYWORDS

st.set_page_config(page_title="Car Inventory Optimization Tool", layout="wide")
st.title("🚗 Smart Car Inventory Optimization Tool (with Real-time & ML Forecasting)")

# -----------------------
# Upload dataset or fallback
# -----------------------
st.sidebar.header("📁 Upload Your Inventory Data")
uploaded_file = st.sidebar.file_uploader("Upload merged car inventory CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    try:
        df = pd.read_csv("final_merged_car_inventory_enhanced.csv")
        st.info("📦 Using default dataset (final_merged_car_inventory_enhanced.csv)")
    except:
        st.error("❌ Please upload a dataset to continue.")
        st.stop()

# -----------------------
# RAW PREVIEW
# -----------------------
st.subheader("📊 Raw Inventory Data Preview")
st.dataframe(df.head(10))

# -----------------------
# CITY-WISE DEMAND MAP
# -----------------------
st.subheader("🌍 City-wise Demand Map")
city_demand = df.groupby("City").agg(
    demand=("DemandScore", "mean"),
    car_count=("Car_Name", "count")
).reset_index()

fig = px.bar(city_demand, x="City", y="demand", color="car_count",
             title="Average Demand Score per City")
st.plotly_chart(fig, use_container_width=True)

# -----------------------
# GOOGLE TRENDS DEMAND
# -----------------------
from api_utils import get_trend_score_all, ALL_TREND_KEYWORDS

st.subheader("📈 Real-time Regional Demand Score (Google Trends - auto keywords)")

trend_df = get_trend_score_all(ALL_TREND_KEYWORDS)

if not trend_df.empty:
    st.dataframe(trend_df.head(10))
else:
    st.warning("⚠️ Google Trends returned no demand data. Showing fallback...")

    if "DemandScore" not in df.columns or df["DemandScore"].isnull().all():
        df["DemandScore"] = np.random.randint(50, 100, size=len(df))

    fallback_df = df.groupby("City").agg(
        AvgDemandScore=("DemandScore", "mean"),
        car_count=("Car_Name", "count")
    ).reset_index()

    st.dataframe(fallback_df.sort_values("AvgDemandScore", ascending=False).head(10))

# -----------------------
# PURCHASE SUGGESTIONS
# -----------------------
st.subheader("🛒 Purchase Suggestions Based on Market Gaps")
if "Base_Model" in df.columns:
    car_demand = df.groupby(["City", "Base_Model"])["DemandScore"].mean().reset_index(name="demand")
    car_supply = df.groupby(["City", "Base_Model"]).size().reset_index(name="supply")
    car_market = pd.merge(car_demand, car_supply, on=["City", "Base_Model"])
    car_market["gap"] = car_market["demand"] - car_market["supply"]
    st.dataframe(car_market[car_market["gap"] > 0].sort_values("gap", ascending=False))

# -----------------------
# ML DEMAND FORECASTING
# -----------------------
st.subheader("🔮 ML-Based Demand Forecasting")
df["past_demand"] = df["DemandScore"] + np.random.randint(-10, 10, size=len(df))
df["days_on_platform"] = np.random.randint(10, 90, size=len(df))

features = ["past_demand", "days_on_platform"]
X = df[features]
y = df["DemandScore"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
df["projected_demand"] = model.predict(X)
st.dataframe(df[["Car_Name", "City", "past_demand", "days_on_platform", "projected_demand"]].head(10))
