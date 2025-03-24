#This script is the first pass of adding eircodes to the dataset. IT uses nominatim to find the eircodes. This is free and unlimited but is rate limited and the data is incomplete.
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


df = pd.read_csv(os.getenv("NewHousePath"))
output_file = os.getenv("NewHousePath")

#Check if file already exists and resume from where it left off
if os.path.exists(output_file):
    existing_df = pd.read_csv(output_file)
    
    if "eircode" in existing_df.columns:
        df["eircode"] = existing_df["eircode"]
    else:
        df["eircode"] = None
else:
    df["eircode"] = None  

geolocator = Nominatim(user_agent="eircode_finder")

# Use nomatim to get eircode
def get_eircode(lat, lon, retries=3):
    """Get eircode from latitude & longitude, retrying if needed."""
    for attempt in range(retries):
        try:
            location = geolocator.reverse((lat, lon), exactly_one=True, addressdetails=True)
            if location and 'postcode' in location.raw['address']:
                return location.raw['address']['postcode']
        except GeocoderTimedOut:
            print(f"Timeout for ({lat}, {lon}) - Retrying ({attempt + 1}/{retries})...")
            time.sleep(2)
        except Exception as e:
            print(f"Error: {e}")
            return None  
    
    return None  

# Resume from last row processed
start_index = df[df["eircode"].isna()].index.min()

# Process rows
if start_index is None:
    print("All rows already processed! ✅")
else:
    print(f"Resuming from row {start_index}...")

    for i in tqdm(range(start_index, len(df)), desc="Processing"):
        if pd.isna(df.at[i, "eircode"]):  
            df.at[i, "eircode"] = get_eircode(df.at[i, "latitude"], df.at[i, "longitude"])

        if i % 2000 == 0 or i == len(df) - 1:
            df.to_csv(output_file, index=False)
            print(f"✅ Saved progress at row {i}.")

    df.to_csv(output_file, index=False)
    print("🎉 Processing complete. Final file saved.")
