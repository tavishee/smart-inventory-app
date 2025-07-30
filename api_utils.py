import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Car Inventory Optimization Tool", layout="wide")
st.title("🚗 Smart Car Inventory Optimization Tool (with ML Forecasting)")

st.sidebar.header("📁 Upload Your Inventory Data")
uploaded_file = st.sidebar.file_uploader("Upload merged car inventory CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Inventory Data Preview")
    st.dataframe(df.head(10))

    # -------------------
    # DEMAND MAP
    # -------------------
    st.subheader("🌍 City-wise Demand Map")
    city_demand = df.groupby("City").agg(
        demand=("DemandScore", "mean"),
        car_count=("Car_Name", "count")
    ).reset_index()

    fig = px.bar(city_demand, x="City", y="demand", color="car_count",
                 title="Average Demand Score per City")
    st.plotly_chart(fig, use_container_width=True)

    # -------------------
    # INVENTORY RISK
    # -------------------
    st.subheader("⚠️ Inventory Risk Dashboard")
    if "days_in_inventory" not in df.columns:
        df["days_in_inventory"] = np.random.randint(20, 150, size=len(df))

    st.dataframe(df[["City", "Car_Name", "days_in_inventory"]].sort_values("days_in_inventory", ascending=False))

    # -------------------
    # RELOCATION ENGINE
    # -------------------
    st.subheader("🚚 Smart Relocation Profit Suggestions")
    demand_by_city = df.groupby("City")["DemandScore"].mean().reset_index(name="avg_demand")
    supply_by_city = df.groupby("City")["Car_Name"].count().reset_index(name="supply")

    combined = pd.merge(demand_by_city, supply_by_city, on="City")
    combined["demand_gap"] = combined["avg_demand"] - combined["supply"]

    surplus_cities = combined[combined["demand_gap"] < 0].copy()
    deficit_cities = combined[combined["demand_gap"] > 0].copy()

    relocation_suggestions = []
    for _, source in surplus_cities.iterrows():
        for _, dest in deficit_cities.iterrows():
            if source["City"] != dest["City"]:
                distance_km = random.randint(100, 2000)
                profit_margin = (dest["avg_demand"] - source["avg_demand"]) * 100
                transport_cost = distance_km * 5
                expected_profit = profit_margin - transport_cost

                if expected_profit > 0:
                    relocation_suggestions.append({
                        "source_city": source["City"],
                        "dest_city": dest["City"],
                        "distance_km": distance_km,
                        "expected_profit": round(expected_profit, 2)
                    })

    reloc_df = pd.DataFrame(relocation_suggestions)
    st.markdown("**Top 10 Profitable Relocation Moves**")
    st.dataframe(reloc_df.sort_values("expected_profit", ascending=False).head(10))

    # -------------------
    # PURCHASE SUGGESTIONS
    # -------------------
    st.subheader("🛒 Purchase Suggestions Based on Market Gaps")
    if "Base_Model" in df.columns:
        car_demand = df.groupby(["City", "Base_Model"])["DemandScore"].mean().reset_index(name="demand")
        car_supply = df.groupby(["City", "Base_Model"]).size().reset_index(name="supply")
        car_market = pd.merge(car_demand, car_supply, on=["City", "Base_Model"])
        car_market["gap"] = car_market["demand"] - car_market["supply"]
        st.dataframe(car_market[car_market["gap"] > 0].sort_values("gap", ascending=False))

    # -------------------
    # ML DEMAND FORECASTING
    # -------------------
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

    # -------------------
    # ML PRICE OPTIMIZATION
    # -------------------
    st.subheader("💰 ML-Based Price Optimization")
    df = df.dropna(subset=["Selling_Price", "Present_Price", "DemandScore"])
    price_features = ["Present_Price", "DemandScore", "days_on_platform"]
    Xp = df[price_features]
    yp = df["Selling_Price"]

    Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, yp, test_size=0.2, random_state=42)
    price_model = LinearRegression()
    price_model.fit(Xp_train, yp_train)

    df["optimal_price"] = price_model.predict(Xp)
    st.dataframe(df[["Car_Name", "City", "Selling_Price", "optimal_price"]].head(10))

else:
    st.warning("⚠️ Please upload your inventory CSV file to proceed.")
