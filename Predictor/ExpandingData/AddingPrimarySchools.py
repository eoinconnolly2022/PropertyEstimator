# This script adds the distance to the nearest mainstream and special primary school to the dataset

import pandas as pd
from sklearn.neighbors import BallTree
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

# Load and preprocess datasets

houses_df = pd.DataFrame(pd.read_csv(os.getenv("NewHousePath")))
mainstream_df = pd.DataFrame(pd.read_csv(os.getenv("MainstreamPath")))
special_df = pd.DataFrame(pd.read_csv(os.getenv("SpecialPath")))

houses_df = houses_df.dropna(subset=['latitude', 'longitude'])
mainstream_df = mainstream_df.dropna(subset=['Latitude', 'Longitude'])
special_df = special_df.dropna(subset=['Latitude', 'Longitude'])

houses_df['latitude'] = pd.to_numeric(houses_df['latitude'], errors='coerce')
houses_df['longitude'] = pd.to_numeric(houses_df['longitude'], errors='coerce')
mainstream_df['Latitude'] = pd.to_numeric(mainstream_df['Latitude'], errors='coerce')
mainstream_df['Longitude'] = pd.to_numeric(mainstream_df['Longitude'], errors='coerce')
special_df['Latitude'] = pd.to_numeric(special_df['Latitude'], errors='coerce')
special_df['Longitude'] = pd.to_numeric(special_df['Longitude'], errors='coerce')

houses_df = houses_df.dropna(subset=['latitude', 'longitude'])
mainstream_df = mainstream_df.dropna(subset=['Latitude', 'Longitude'])
special_df = special_df.dropna(subset=['Latitude', 'Longitude'])

# calculate the nearest distance to the nearest mainstream and special primary school using ball tree and haversine distance
def calculate_nearest_distance_fast(houses_df, schools_df, column_name):
    houses_coords = np.radians(houses_df[['latitude', 'longitude']].values)
    schools_coords = np.radians(schools_df[['Latitude', 'Longitude']].values)
    tree = BallTree(schools_coords, metric='haversine')
    distances, _ = tree.query(houses_coords, k=1)
    distances_km = distances[:, 0] * 6371
    houses_df[column_name] = distances_km
    return houses_df

houses_df = calculate_nearest_distance_fast(houses_df, mainstream_df, 'nearest_mainstream_distance')
houses_df = calculate_nearest_distance_fast(houses_df, special_df, 'nearest_special_distance')

houses_df.to_csv(os.getenv("NewHousePath"), index=False, encoding="utf-8")

