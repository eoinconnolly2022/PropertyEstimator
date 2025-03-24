import pandas as pd
from dotenv import load_dotenv
import os
load_dotenv()
df = pd.DataFrame(pd.read_csv(os.getenv("NewHousePath")))

columns_to_string = ['property_type']

df = df.fillna('None')
for column in columns_to_string:
    df[column] = df[column].astype(str)

print(df.dtypes)