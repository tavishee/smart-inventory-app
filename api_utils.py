from pytrends.request import TrendReq
import pandas as pd
import time

ALL_TREND_KEYWORDS = [
    "used car", "second hand car", "pre owned car", "cheap car", "car under 5 lakhs",
    "buy car", "car EMI", "used car loan", "certified used car", "car resale value",
    "Maruti", "Maruti Swift", "Maruti Alto", "Maruti Baleno", "Wagon R", "Ertiga",
    "Hyundai", "Hyundai i20", "Hyundai Creta", "Venue", "Santro",
    "Tata", "Tata Nexon", "Tata Tiago", "Tata Punch",
    "Honda", "Honda City", "Amaze",
    "Toyota", "Toyota Innova", "Toyota Fortuner", "Etios",
    "Kia", "Kia Seltos", "Kia Sonet",
    "Mahindra", "Mahindra XUV300", "Scorpio", "Bolero", "Thar",
    "Renault", "Renault Kwid", "Triber",
    "Volkswagen", "Skoda", "Ford EcoSport", "Nissan Magnite",
    "SUV", "sedan", "hatchback", "7 seater car", "family car",
    "compact SUV", "CNG car", "automatic car", "manual car",
    "second hand car in Delhi", "second hand Swift", "used car Bangalore",
    "best used car", "cheap Alto", "Maruti second hand", "car resale", "second hand SUV India"
]

def get_trend_score_all(keywords):
    pytrends = TrendReq(hl='en-US', tz=330)
    results = []

    for keyword in keywords:
        try:
            pytrends.build_payload([keyword], cat=0, timeframe='now 7-d', geo='IN')
            data = pytrends.interest_by_region(resolution='REGION', inc_low_vol=True)
            data = data[[keyword]].reset_index()
            data.columns = ["Region", "Score"]
            data["Keyword"] = keyword
            results.append(data)
            time.sleep(1)
        except Exception as e:
            print(f"[Trend fail for '{keyword}']: {e}")
            continue

    if not results:
        return pd.DataFrame()

    all_data = pd.concat(results)
    grouped = all_data.groupby("Region")["Score"].mean().reset_index()
    grouped.columns = ["Region", "AvgDemandScore"]
    return grouped.sort_values("AvgDemandScore", ascending=False)
