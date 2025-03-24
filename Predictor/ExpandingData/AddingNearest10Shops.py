# This script adds the average distance to the nearest ten shops to the dataset

import pandas as pd
from sklearn.neighbors import BallTree
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

# Load datasets
houses_df = pd.DataFrame(pd.read_csv(os.getenv("NewHousePath")))
shops_df = pd.DataFrame(pd.read_csv(os.getenv("ShopsPath")))

# calculate the average distance to the nearest 10 shops using ball tree and haversine distance
def calculate_average_distance_fast(houses_df, shops_df, n=10):
    houses_coords = np.radians(houses_df[['latitude', 'longitude']].values)
    shops_coords = np.radians(shops_df[['latitude', 'longitude']].values)
    tree = BallTree(shops_coords, metric='haversine')
    distances, _ = tree.query(houses_coords, k=n)
    distances_km = distances * 6371
    houses_df['avg_dist_to_10_shops'] = distances_km.mean(axis=1)
    return houses_df

houses_df = calculate_average_distance_fast(houses_df, shops_df, n=5)

houses_df.to_csv(os.getenv("NewHousePath"), index=False, encoding="utf-8")
