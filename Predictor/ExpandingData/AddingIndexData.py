#This script adds the deprivation index data, along with other data from the dataset to the housing dataset by finding the nearest region to each property.

import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree


import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load datasets
houses = pd.read_csv("eircode_dataset.csv")
regions = pd.read_csv(os.getenv("ProsperityCoordsPath"))

# Extract coordinates for BallTree
region_coords = regions[['latitude', 'longitude']].to_numpy()
house_coords = houses[['latitude', 'longitude']].to_numpy()

# Create a BallTree for efficient nearest neighbor search
tree = BallTree(np.radians(region_coords), metric='haversine')  # Use haversine distance for geo-coordinates

# Find nearest region for each house
_, indices = tree.query(np.radians(house_coords), k=1)
nearest_regions = regions.iloc[indices.flatten()].reset_index(drop=True)

# Drop coordinate columns from region data
numerical_columns = regions.select_dtypes(include=[np.number]).columns.tolist()
numerical_columns.remove('latitude')
numerical_columns.remove('longitude')

# Merge nearest region data into house dataset
houses = pd.concat([houses, nearest_regions[numerical_columns].add_prefix('region_')], axis=1)

print(houses)
# Save the merged dataset
houses.to_csv(os.getenv("NewHousePath"), index=False)

