# Script used to check what amount of the properties' eircodes were missed by the reverse geocoding process
import pandas as pd

df = pd.DataFrame(pd.read_csv("eircode_dataset.csv"))

print(df.head())

print(df.tail())

print(df['eircode'].isnull().sum())