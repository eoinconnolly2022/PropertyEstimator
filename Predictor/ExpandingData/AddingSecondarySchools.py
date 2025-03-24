# This script adds the distance to the nearest secondary school to the dataset

import pandas as pd
from sklearn.neighbors import BallTree
import numpy as np
from dotenv import load_dotenv
import os
load_dotenv()

#Load and preprocess datasets
houses_df = pd.DataFrame(pd.read_csv(os.getenv("NewHousePath")))
secondary_df = pd.DataFrame(pd.read_csv(os.getenv("SecondaryPath")))

houses_df = houses_df.dropna(subset=['latitude', 'longitude'])
secondary_df = secondary_df.dropna(subset=['Latitude', 'Longitude'])

houses_df['latitude'] = pd.to_numeric(houses_df['latitude'], errors='coerce')
houses_df['longitude'] = pd.to_numeric(houses_df['longitude'], errors='coerce')
secondary_df['Latitude'] = pd.to_numeric(secondary_df['Latitude'], errors='coerce')
secondary_df['Longitude'] = pd.to_numeric(secondary_df['Longitude'], errors='coerce')

houses_df = houses_df.dropna(subset=['latitude', 'longitude'])
secondary_df = secondary_df.dropna(subset=['Latitude', 'Longitude'])

# calculate the nearest distance to the nearest secondary school using ball tree and haversine distance
def calculate_nearest_distance_fast(houses_df, schools_df, column_name):
    houses_coords = np.radians(houses_df[['latitude', 'longitude']].values)
    schools_coords = np.radians(schools_df[['Latitude', 'Longitude']].values)
    tree = BallTree(schools_coords, metric='haversine')
    distances, _ = tree.query(houses_coords, k=1)
    distances_km = distances[:, 0] * 6371
    houses_df[column_name] = distances_km
    return houses_df

houses_df = calculate_nearest_distance_fast(houses_df, secondary_df, 'nearest_secondary_distance')

#save the dataset
houses_df.to_csv(os.getenv("NewHousePath"), index=False, encoding="utf-8")

