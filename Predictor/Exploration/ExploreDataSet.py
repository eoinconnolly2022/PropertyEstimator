import pandas as pd
from dotenv import load_dotenv
import os
load_dotenv()
df = pd.DataFrame(pd.read_csv(os.getenv("NewHousePath")))


print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.columns)
print(df.shape)
print(df.isnull().sum())
print(df.dtypes)
