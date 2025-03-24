# This script simply adds a placeholder to the property type column as some entries are unspecified

import pandas as pd

df = pd.DataFrame(pd.read_csv("eircode_dataset.csv"))

df['property_type'] = df['property_type'].fillna('Unspecified')

df.to_csv("eircode_dataset_with_placeholder.csv", index=False)