#This script is a second reverse geocoding pass. Nomatim which was used in the first pass has a lot of missing data, however, better APIs are expensive. This script uses the HERE API which is free for a limited number of requests. The script will resume from where it left off and save progress every 10 eircodes found.
import pandas as pd
import requests
import time
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

# Load HERE API Key
HERE_API_KEY = os.getenv("HERE_API_KEY")
HERE_REVERSE_API_URL = "https://revgeocode.search.hereapi.com/v1/revgeocode"

# Load existing output file
temp_df = pd.read_csv(os.getenv("NewHousePath"))
output_file = os.getenv("NewHousePath")

#Usi
def get_eircode(lat, lon, retries=3):
    """Get eircode from latitude & longitude using HERE Maps API, retrying if needed."""
    for attempt in range(retries):
        try:
            params = {
                'at': f"{lat},{lon}",
                'apiKey': HERE_API_KEY
            }
            response = requests.get(HERE_REVERSE_API_URL, params=params)
            
            if response.status_code != 200:
                print(f"Error reaching HERE API: {response.text}")
                return None

            results = response.json()
            
            if 'items' not in results or len(results['items']) == 0:
                return None
            
            address = results['items'][0]['address']
            return address.get('postalCode', None)
        
        except Exception as e:
            print(f"Error: {e} - Retrying ({attempt + 1}/{retries})...")
            time.sleep(2)
    
    return None

# Start from the first row where 'eircode' is missing
start_index = temp_df[temp_df["eircode"].isna()].index.min()

if start_index is None:
    print("All rows already processed! ✅")
else:
    print(f"Resuming from row {start_index}...")

    eircodes_found = 0
    for i in tqdm(range(start_index, len(temp_df)), desc="Processing"):
        if pd.isna(temp_df.at[i, "eircode"]):
            eircode = get_eircode(temp_df.at[i, "latitude"], temp_df.at[i, "longitude"])
            if eircode:
                temp_df.at[i, "eircode"] = eircode
                print(f"Found Eircode: {eircode} at row {i}")
                eircodes_found += 1

        # Save every 10 eircodes found
        if eircodes_found >= 10:
            temp_df.to_csv(output_file, index=False)
            print(f"✅ Saved progress after finding 10 eircodes.")
            eircodes_found = 0

    # Final save
    temp_df.to_csv(output_file, index=False)
    print("🎉 Processing complete. Final file saved.")
