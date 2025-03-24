# This script takes the original excel file and removes any null rows and converts to a csv for easier processing.

import pandas as pd

df = pd.DataFrame(pd.read_excel("/mnt/c/Users/eoinc/Desktop/House price/HouseEstimator/Data/OriginalData/FullHousePriceDataSet.xlsx"))


df_cleaned = df.dropna(how='all')


df_cleaned.to_csv("/mnt/c/Users/eoinc/Desktop/House price/HouseEstimator/Data/ExpandedData/CleanedHousePriceDataSet.csv", index=False)


print("DataFrame saved as CSV successfully.")
