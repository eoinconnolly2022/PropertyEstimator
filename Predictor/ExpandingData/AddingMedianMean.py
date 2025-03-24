# This script adds the Median and Mean prices from December 2024 to the house dataset
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load datasets
df = pd.read_csv("eircode_dataset.csv")
df2 = pd.read_csv(os.getenv("MedianMeanPath"))

# Extract first three characters of eircode
df['eircodeThree'] = df['eircode'].str[:3]
df2['eircode1'] = df2['Eircode Output'].str[:3]

# Pivot Mean and Median table so that Median and Mean prices are in the same row
df2_pivoted = df2.pivot(index='eircode1', columns='Statistic Label', values='VALUE').reset_index()

# Rename columns for clarity
df2_pivoted.columns = ['eircode1', 'Mean Sale Price', 'Median Sale Price']  # Adjust names as needed

# Merge with the house dataset
merged_df = df.merge(df2_pivoted, left_on='eircodeThree', right_on='eircode1', how='left')

# Drop unnecessary columns
merged_df = merged_df.drop(columns=['eircodeThree', 'eircode1'])

# Print the merged dataset
print(merged_df)


merged_df.to_csv(os.getenv('NewHousePath'), index=False, encoding="utf-8")