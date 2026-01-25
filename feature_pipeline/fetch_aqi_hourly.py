# ================= feature_pipeline/fetch_aqi_hourly.py =================
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv() 

# ---------------- CONFIG ----------------
AQICN_TOKEN = "59741dd6dd39e39a9380da6133bc2f0fe1656336"
CITY = "Karachi"
AQI_URL = f"https://api.waqi.info/feed/Karachi/?token=59741dd6dd39e39a9380da6133bc2f0fe1656336"

LAT, LON = 24.8607, 67.0011
WEATHER_URL = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,precipitation"

# ---------------- FETCH LIVE DATA ----------------
def fetch_live_data():
    # 1️⃣ AQI data
    aqi_resp = requests.get(AQI_URL).json()
    iaqi = aqi_resp['data']['iaqi']
    aqi_row = {
        "timestamp": pd.to_datetime(aqi_resp['data']['time']['iso']),
        "aqi": aqi_resp['data']['aqi'],
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "so2": iaqi.get("so2", {}).get("v"),
        "co": iaqi.get("co", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v")
    }
    aqi_df = pd.DataFrame([aqi_row])

    # Convert AQI timestamp to timezone-naive
    aqi_df['timestamp'] = aqi_df['timestamp'].dt.tz_convert(None)

    # 2️⃣ Weather data
    weather_resp = requests.get(WEATHER_URL).json()
    weather_df = pd.DataFrame(weather_resp['hourly'])
    weather_df['timestamp'] = pd.to_datetime(weather_df['time']).dt.tz_localize(None)
    weather_df = weather_df.drop(columns=['time'])

    # 3️⃣ Merge AQI + weather
    merged_df = pd.merge_asof(
        aqi_df.sort_values("timestamp"),
        weather_df.sort_values("timestamp"),
        on="timestamp",
        direction="nearest"
    )

    return merged_df

# ---------------- MAIN ----------------
if __name__ == "__main__":
    df = fetch_live_data()
    print("✅ Fetched live AQI + weather data:")
    print(df.head())
