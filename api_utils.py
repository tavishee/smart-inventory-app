import googlemaps
from pytrends.request import TrendReq
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Google Maps Distance
def get_distance_km(origin, destination):
    try:
        gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))
        result = gmaps.distance_matrix(origins=[origin], destinations=[destination], units="metric")
        distance = result["rows"][0]["elements"][0]["distance"]["value"] / 1000  # in km
        return round(distance, 2)
    except Exception as e:
        print(f"[Google Maps Error] {e}")
        return None

from pytrends.request import TrendReq
import pandas as pd

def get_trend_score(keyword):
    try:
        pytrends = TrendReq(hl='en-US', tz=330)
        pytrends.build_payload([keyword], cat=0, timeframe='now 7-d', geo='IN')
        data = pytrends.interest_by_region(resolution='REGION', inc_low_vol=True)
        top_regions = data.sort_values(by=keyword, ascending=False).reset_index()
        return top_regions
    except Exception as e:
        print(f"[Trend error] {e}")
        return pd.DataFrame()


# 3. Fuel Price (simulated for now)
def get_fuel_price(city):
    fuel_prices = {
        "Delhi": 96.7,
        "Mumbai": 104.8,
        "Bangalore": 101.1,
        "Hyderabad": 100.4,
        "Kolkata": 103.5,
        "Chennai": 102.3
    }
    return fuel_prices.get(city, 95.0)
