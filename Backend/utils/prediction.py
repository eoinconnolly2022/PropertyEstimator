from http.client import HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import requests
import re
from sklearn.neighbors import BallTree
import psycopg2
from psycopg2 import sql
import os
from dotenv import load_dotenv
load_dotenv()

# Database config
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),  
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

# Load and preprocess data
median_prices_df = pd.read_csv('Data/AveragePrices.csv')
median_prices_dict = {row['Eircode Output'][:3]: row['VALUE'] for _, row in median_prices_df.iterrows()}
index_prices_df = pd.read_csv('Data/RPPI.csv')
index_prices_df['region'] = index_prices_df['Type of Residential Property'].str.split(' - ').str[0]

# Initialize BallTrees for all distance calculations
shops_df = pd.read_csv('Data/shops_ireland.csv')
shops_coords_rad = np.deg2rad(shops_df[['latitude', 'longitude']].values)
shops_tree = BallTree(shops_coords_rad, metric='haversine')

mainstream_df = pd.read_csv('Data/mainstream_schools.csv')
mainstream_df['Latitude'] = pd.to_numeric(mainstream_df['Latitude'], errors='coerce')
mainstream_df['Longitude'] = pd.to_numeric(mainstream_df['Longitude'], errors='coerce')
mainstream_df = mainstream_df.dropna(subset=['Latitude', 'Longitude'])
mainstream_coords_rad = np.deg2rad(mainstream_df[['Latitude', 'Longitude']].values)
mainstream_tree = BallTree(mainstream_coords_rad, metric='haversine')

special_df = pd.read_csv('Data/special_schools.csv')
special_df['Latitude'] = pd.to_numeric(special_df['Latitude'], errors='coerce')
special_df['Longitude'] = pd.to_numeric(special_df['Longitude'], errors='coerce')
special_df = special_df.dropna(subset=['Latitude', 'Longitude'])
special_coords_rad = np.deg2rad(special_df[['Latitude', 'Longitude']].values)
special_tree = BallTree(special_coords_rad, metric='haversine')

secondary_df = pd.read_csv('Data/secondary_schools.csv')
secondary_df['Latitude'] = pd.to_numeric(secondary_df['Latitude'], errors='coerce')
secondary_df['Longitude'] = pd.to_numeric(secondary_df['Longitude'], errors='coerce')
secondary_df = secondary_df.dropna(subset=['Latitude', 'Longitude'])
secondary_coords_rad = np.deg2rad(secondary_df[['Latitude', 'Longitude']].values)
secondary_tree = BallTree(secondary_coords_rad, metric='haversine')

prosperity_df = pd.read_csv('Data/merged_prosperity.csv')
prosperity_coords_rad = np.deg2rad(prosperity_df[['latitude', 'longitude']].values)
prosperity_tree = BallTree(prosperity_coords_rad, metric='haversine')

# Helper function for PostgreSQL queries
def execute_query(query, params=None, fetch=False):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    try:
        cursor.execute(query, params)
        if fetch:
            result = cursor.fetchall()
        else:
            result = None
        conn.commit()
        return result
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        conn.rollback()
        return None
    finally:
        cursor.close()
        conn.close()

# Dictionary to map eircode prefixes to regions
eircode_to_region = {
    "D01": "Dublin City", "D02": "Dublin City", "D03": "Dublin City", "D04": "Dublin City", "D05": "Dublin City",
    "D06": "Dublin City", "D6W": "Dublin City", "D07": "Dublin City", "D08": "Dublin City", "D09": "Dublin City",
    "D10": "Dublin City", "D11": "Fingal", "D12": "Dublin City", "D13": "Dublin City", "D14": "Dun Laoghaire-Rathdown",
    "D15": "Fingal", "D16": "Dun Laoghaire-Rathdown", "D17": "Dublin City", "D18": "Dun Laoghaire-Rathdown",
    "D20": "South Dublin", "D22": "South Dublin", "D24": "South Dublin", "A92": "Mid-East", "Y14": "Mid-East",
    "A84": "Mid-East", "H65": "West", "N37": "Midland", "R14": "Mid-East", "K32": "Fingal", "F26": "West",
    "H53": "West", "P31": "South-West", "F31": "West", "A75": "Border", "A41": "Fingal", "F35": "West",
    "F56": "Border", "P72": "South-West", "P75": "South-West", "H14": "Border", "R42": "Midland", "A94": "Dun Laoghaire-Rathdown",
    "F52": "West", "A98": "Mid-East", "V23": "South-West", "E21": "Mid-West", "R93": "South-East", "A81": "Border",
    "N41": "Border", "E32": "Mid-West", "P43": "South-West", "E25": "Mid-West", "F23": "West", "F45": "West",
    "H12": "Border", "P56": "South-West", "F12": "West", "H71": "West", "P85": "South-West", "H23": "Border",
    "E91": "Mid-West", "P24": "South-West", "H16": "Border", "T12": "South-West", "T23": "South-West", "P14": "South-West",
    "P32": "South-West", "P47": "South-West", "T56": "South-West", "T34": "South-West", "R56": "Mid-East", "A63": "Mid-East",
    "F94": "Border", "A86": "Mid-East", "A91": "Mid-East", "X35": "South-East", "A85": "Mid-East", "R45": "Mid-East",
    "A83": "Mid-East", "V95": "Mid-West", "Y21": "South-East", "P61": "South-West", "H91": "West", "A42": "Fingal",
    "A96": "Dun Laoghaire-Rathdown", "Y25": "South-East", "A63": "Mid-East", "A82": "Mid-East", "R51": "Mid-East",
    "R95": "South-East", "V93": "South-West", "X42": "South-East", "V35": "Mid-West", "V15": "Mid-West", "A82": "Border",
    "P17": "South-West", "F92": "Border", "F93": "Border", "V94": "Mid-West", "V31": "South-West", "T45": "South-West",
    "N39": "Midland", "H62": "West", "K78": "South Dublin", "K45": "Fingal", "P12": "South-West", "K36": "Fingal",
    "P51": "South-West", "W23": "Mid-East", "P25": "South-West", "P67": "South-West", "H18": "Border", "W34": "Mid-East",
    "R21": "South-East", "N91": "Midland", "W91": "Mid-East", "C15": "Mid-East", "E45": "Mid-West", "Y34": "South-East",
    "W12": "Mid-East", "V42": "Mid-West", "A45": "Fingal", "R32": "Midland", "A67": "Mid-East", "F42": "West",
    "E53": "Mid-West", "K56": "Fingal", "V14": "Mid-West", "K34": "Fingal", "P81": "South-West", "F91": "Border",
    "A83": "Mid-East", "K67": "Fingal", "E41": "Mid-West", "E34": "Mid-West", "V92": "South-West", "H54": "West",
    "R35": "Midland", "A82": "Border", "X91": "South-East", "F28": "West", "Y35": "South-East", "A67": "Mid-East",
    "P36": "South-West"
}

# API key for geocoding service
API_KEY = os.getenv("HERE_API_KEY")
HERE_API_URL = "https://geocode.search.hereapi.com/v1/geocode"

# Function to validate eircode format
def validate_eircode(eircode: str):
    if not re.match(r"^[A-Z0-9]{3}\s?[A-Z0-9]{4}$", eircode.upper()):
        raise HTTPException(status_code=400, detail="Invalid Eircode format")

# Geocoding function
def get_lat_long_from_eircode(eircode: str):
    validate_eircode(eircode)

    params = {
        'q': eircode,
        'apiKey': API_KEY
    }
    response = requests.get(HERE_API_URL, params=params)
    
    if response.status_code != 200:
        raise HTTPException(status_code=503, detail=f"Error reaching geocoding service: {response.text}")

    results = response.json()

    if 'items' not in results or len(results['items']) == 0:
        raise HTTPException(status_code=404, detail="Eircode not found")

    latitude = results['items'][0]['position']['lat']
    longitude = results['items'][0]['position']['lng']
    
    return latitude, longitude

# Function to remove end of eircode as it is unnecessary for data expansion
def extract_eircode_prefix(eircode: str) -> str:
    return eircode[:3].upper()

# Function to get median and mean sale prices for a given eircode region
def get_median_and_mean_sale_price(eircode_prefix: str) -> tuple[float, float]:
    filtered_df = median_prices_df[median_prices_df["Eircode Output"].str.startswith(eircode_prefix)]

    if filtered_df.empty:
        return None, None

    median_sale_price = filtered_df[filtered_df["Statistic Label"] == "Median Price"]["VALUE"].values[0]
    mean_sale_price = filtered_df[filtered_df["Statistic Label"] == "Mean Sale Price"]["VALUE"].values[0]

    median_sale_price = float(median_sale_price) if median_sale_price else None
    mean_sale_price = float(mean_sale_price) if mean_sale_price else None

    return median_sale_price, mean_sale_price

# Function to get RPPI value for a given eircode region and date
def get_RPPI_from_eircode_date(eircode_prefix: str, date: str) -> float:
    region = eircode_to_region.get(eircode_prefix)
    
    if not region:
        raise HTTPException(status_code=404, detail="Region not found for the given eircode prefix")

    filtered_df = index_prices_df[index_prices_df['region'] == region].copy()
    filtered_df.loc[:, 'Month'] = pd.to_datetime(filtered_df['Month'], format='%Y %B')
    target_date = pd.to_datetime(date, format='%d/%m/%Y')

    if target_date < filtered_df['Month'].min():
        closest_row = filtered_df.loc[filtered_df['Month'] == filtered_df['Month'].min()]
    elif target_date > filtered_df['Month'].max():
        closest_row = filtered_df.loc[filtered_df['Month'] == filtered_df['Month'].max()]
    else:
        closest_row = filtered_df.iloc[(filtered_df['Month'] - target_date).abs().argsort()[:1]]

    return float(closest_row['VALUE'].iloc[0]) if not closest_row.empty else None

# Function to get nearest prosperity index values using BallTree
def get_nearest_index_value(latitude: float, longitude: float):
    house_coord_rad = np.deg2rad(np.array([[latitude, longitude]]))
    distances_rad, indices = prosperity_tree.query(house_coord_rad, k=1)
    closest_row = prosperity_df.iloc[indices[0][0]]
    closest_row['distance'] = distances_rad[0][0] * 6371  # Convert to km
    return closest_row

# Function to calculate average distance to n nearest shops using BallTree
def calculate_average_distance_fast(latitude, longitude, n=10):
    house_coord_rad = np.deg2rad(np.array([[latitude, longitude]]))
    distances_rad, _ = shops_tree.query(house_coord_rad, k=n)
    distances_km = distances_rad[0] * 6371  # Convert to km
    return float(np.mean(distances_km))

# Function to calculate nearest distance using BallTree
def calculate_nearest_distance_fast(latitude, longitude, tree):
    house_coord_rad = np.deg2rad(np.array([[latitude, longitude]]))
    distances_rad, _ = tree.query(house_coord_rad, k=1)
    return float(distances_rad[0][0] * 6371)  # Convert to km

# Dictionary to map property types to categories. Property types taken from the dataset. Default = houses
house_to_type = {
    "House": "houses", "Detached House": "houses", "Semi-Detached House": "houses", "Terraced House": "houses", 
    "End of Terrace": "houses", "Bungalow": "houses", "Townhouse": "houses",
    "Apartment": "apartments", "Duplex": "apartments", "Studio": "apartments",
    "Unspecified": "houses"
}

def fetch_additional_features(eircode: str):
    current_date = datetime.now()
    sold_year = current_date.year
    sold_month = current_date.month
    sold_day = current_date.day

    latitude, longitude = get_lat_long_from_eircode(eircode)

    median_sale_price, mean_sale_price = get_median_and_mean_sale_price(extract_eircode_prefix(eircode))
    
    RPPI_value = get_RPPI_from_eircode_date(extract_eircode_prefix(eircode), f"{sold_day}/{sold_month}/{sold_year}")

    closest_row = get_nearest_index_value(latitude, longitude)
    avg_distance_shop = calculate_average_distance_fast(latitude, longitude)

    nearest_mainstream_school_distance = calculate_nearest_distance_fast(latitude, longitude, mainstream_tree)
    nearest_special_school_distance = calculate_nearest_distance_fast(latitude, longitude, special_tree)
    nearest_secondary_school_distance = calculate_nearest_distance_fast(latitude, longitude, secondary_tree)

    date_listed = datetime(sold_year, sold_month, sold_day)
    date_listed_unix = int(date_listed.timestamp())

    return {
        'latitude': latitude,
        'longitude': longitude,
        'date_listed_unix': date_listed_unix,
        'median_sale_price': median_sale_price,
        'mean_sale_price': mean_sale_price,
        'RPPI_VALUE': RPPI_value,
        'Index22_ED_std_abs_wt': closest_row['Index22_ED_std_abs_wt'],
        'avg_dist_to_10_shops': avg_distance_shop,
        'Nearest_Mainstream_School_Distance': nearest_mainstream_school_distance,
        'Nearest_Special_School_Distance': nearest_special_school_distance,
        'Nearest_Secondary_School_Distance': nearest_secondary_school_distance,
        'region_AGEDEP22': closest_row['AGEDEP22'],
        'region_LONEPA22': closest_row['LONEPA22'],
        'region_EDLOW_22': closest_row['EDLOW_22'],
        'region_EDHIGH22': closest_row['EDHIGH22'],
        'region_HLPROF22': closest_row['HLPROF22'],
        'region_LCLASS22': closest_row['LCLASS22'],
        'region_UNEMPM22': closest_row['UNEMPM22'],
        'region_UNEMPF22': closest_row['UNEMPF22'],
        'region_OWNOCC22': closest_row['OWNOCC22'],
        'region_PRRENT22': closest_row['PRRENT22'],
        'region_LARENT22': closest_row['LARENT22'],
        'region_PEROOM22': closest_row['PEROOM22'],
        'region_Index22_ED_std_rel_wt': closest_row['Index22_ED_std_rel_wt'],
        'region_Index22_ED_std_abs_wt': closest_row['Index22_ED_std_abs_wt'],
        'region_Index22_ED_rel_wt_cat': closest_row['Index22_ED_rel_wt_cat'],
        'closest_row': closest_row
    }

# Load models
xgb_model = joblib.load('models/xgboost_model.pkl')
scaler = joblib.load('models/standard_scaler.pkl')

#Prediction request model
class PredictionRequest(BaseModel):
    api: str
    eircode: str
    metres_squared: float
    bedrooms: int
    bathrooms: int
    ber: str
    property_type: str

# Function to log API usage for user. This can be used to track usage and monitor for abuse
def log_api_usage(api_key_id: int, endpoint: str, request_data: str, response_data: str):
    query = sql.SQL("""
    INSERT INTO UsageData (api_key_id, timestamp, endpoint, request_data, response_data)
    VALUES (%s, %s, %s, %s, %s)
    """)
    timestamp = datetime.now()
    execute_query(query, (api_key_id, timestamp, endpoint, request_data, response_data))

# Prediction function
def predict(request: PredictionRequest):
    api_key = request.api
    api_key_query = sql.SQL("""
    SELECT api_key_id, user_id FROM APIKeys WHERE api_key = %s
    """)
    api_key_result = execute_query(api_key_query, (api_key,), fetch=True)

    if not api_key_result:
        return {"status": "failure", "message": "Invalid API key"}

    api_key_id, user_id = api_key_result[0][0], api_key_result[0][1]

    additional_features = fetch_additional_features(request.eircode)
    closest_row = additional_features['closest_row']

    property_category = house_to_type.get(request.property_type, "houses")

    prop_houses = 1 if property_category == "houses" else 0
    prop_apartments = 1 if property_category == "apartments" else 0

    features = [
        request.bedrooms,
        request.bathrooms,
        request.metres_squared,
        additional_features['date_listed_unix'],
        additional_features['latitude'],
        additional_features['longitude'],
        additional_features['Nearest_Secondary_School_Distance'],
        additional_features['Nearest_Mainstream_School_Distance'],
        additional_features['Nearest_Special_School_Distance'],
        additional_features['avg_dist_to_10_shops'],
        additional_features['RPPI_VALUE'],
        additional_features['mean_sale_price'],
        additional_features['median_sale_price'],
        additional_features['region_AGEDEP22'],
        additional_features['region_LONEPA22'],
        additional_features['region_EDLOW_22'],
        additional_features['region_EDHIGH22'],
        additional_features['region_HLPROF22'],
        additional_features['region_LCLASS22'],
        additional_features['region_UNEMPM22'],
        additional_features['region_UNEMPF22'],
        additional_features['region_OWNOCC22'],
        additional_features['region_PRRENT22'],
        additional_features['region_LARENT22'],
        additional_features['region_PEROOM22'],
        additional_features['region_Index22_ED_std_rel_wt'],
        additional_features['region_Index22_ED_std_abs_wt'],
        additional_features['region_Index22_ED_rel_wt_cat'],
        1 if request.ber == 'A1' else 0,
        1 if request.ber == 'A2' else 0,
        1 if request.ber == 'A3' else 0,
        1 if request.ber == 'B1' else 0,
        1 if request.ber == 'B2' else 0,
        1 if request.ber == 'B3' else 0,
        1 if request.ber == 'C1' else 0,
        1 if request.ber == 'C2' else 0,
        1 if request.ber == 'C3' else 0,
        1 if request.ber == 'D1' else 0,
        1 if request.ber == 'D2' else 0,
        1 if request.ber == 'E1' else 0,
        1 if request.ber == 'E2' else 0,
        1 if request.ber == 'Exempt' else 0,
        1 if request.ber == 'F' else 0,
        1 if request.ber == 'G' else 0,
        prop_apartments,
        prop_houses,
    ]

    X_test = np.array(features).reshape(1, -1)
    X_test_scaled = scaler.transform(X_test)
    final_pred_scaled = xgb_model.predict(X_test_scaled)

    final_pred = np.expm1(final_pred_scaled[0])
    total_predicted_price = final_pred * request.metres_squared

    final_pred = float(final_pred)
    total_predicted_price = float(total_predicted_price)

    response = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "prediction": {
            "predictedPricePerMetreSquared": {"value": final_pred, "unit": "EUR/m²"},
            "totalPredictedPrice": {"value": total_predicted_price, "unit": "EUR"}
        },
        "data": {
            "socioeconomic": {
                "medianSalePrice": {"value": additional_features['median_sale_price'], "unit": "EUR"},
                "deprivationIndex": closest_row['Index22_ED_std_abs_wt'],
            },
            "distances": {
                "averageTo10Shops": {"value": additional_features['avg_dist_to_10_shops'], "unit": "km"},
                "nearestMainstreamSchool": {"value": additional_features['Nearest_Mainstream_School_Distance'], "unit": "km"},
                "nearestSpecialSchool": {"value": additional_features['Nearest_Special_School_Distance'], "unit": "km"},
                "nearestSecondarySchool": {"value": additional_features['Nearest_Secondary_School_Distance'], "unit": "km"}
            }
        }
    }

    log_api_usage(api_key_id, "predict", str(request.dict()), str(response))

    return response
