# This script adds the Residential Property Price Index (RPPI) to the house sales dataset.

import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
df = pd.read_csv(os.getenv("NewHousePath"))
df2 = pd.read_csv(os.getenv("RPPIPath"))

# Dictionary to convert Eircode prefix to region
eircode_to_region = {
    "D01": "Dublin City","D02": "Dublin City","D03": "Dublin City","D04": "Dublin City","D05": "Dublin City","D06": "Dublin City","D6W": "Dublin City",
    "D07": "Dublin City","D08": "Dublin City","D09": "Dublin City","D10": "Dublin City","D11": "Fingal","D12": "Dublin City","D13": "Dublin City",
    "D14": "Dun Laoghaire-Rathdown","D15": "Fingal","D16": "Dun Laoghaire-Rathdown","D17": "Dublin City","D18": "Dun Laoghaire-Rathdown",
    "D20": "South Dublin","D22": "South Dublin","D24": "South Dublin","A92": "Mid-East","Y14": "Mid-East","A84": "Mid-East","H65": "West","N37": "Midland",
    "R14": "Mid-East","K32": "Fingal","F26": "West","H53": "West","P31": "South-West","F31": "West","A75": "Border","A41": "Fingal","F35": "West",
    "F56": "Border","P72": "South-West","P75": "South-West","H14": "Border","R42": "Midland","A94": "Dun Laoghaire-Rathdown","F52": "West","A98": "Mid-East",
    "V23": "South-West","E21": "Mid-West","R93": "South-East","A81": "Border","N41": "Border","E32": "Mid-West","P43": "South-West","E25": "Mid-West",
    "F23": "West","F45": "West","H12": "Border","P56": "South-West","F12": "West","H71": "West","P85": "South-West","H23": "Border","E91": "Mid-West",
    "P24": "South-West","H16": "Border","T12": "South-West","T23": "South-West","P14": "South-West","P32": "South-West","P47": "South-West","T56": "South-West",
    "T34": "South-West","R56": "Mid-East","A63": "Mid-East","F94": "Border","A86": "Mid-East","A91": "Mid-East","X35": "South-East","A85": "Mid-East","R45": "Mid-East",
    "A83": "Mid-East","V95": "Mid-West","Y21": "South-East","P61": "South-West","H91": "West","A42": "Fingal","A96": "Dun Laoghaire-Rathdown","Y25": "South-East",
    "A63": "Mid-East","A82": "Mid-East","R51": "Mid-East","R95": "South-East","V93": "South-West","X42": "South-East","V35": "Mid-West","V15": "Mid-West",
    "A82": "Border","P17": "South-West","F92": "Border", "F93": "Border", "V94": "Mid-West", "V31": "South-West","T45": "South-West","N39": "Midland",
    "H62": "West","K78": "South Dublin","K45": "Fingal","P12": "South-West","K36": "Fingal","P51": "South-West","W23": "Mid-East","P25": "South-West",
    "P67": "South-West","H18": "Border","W34": "Mid-East","R21": "South-East","N91": "Midland","W91": "Mid-East","C15": "Mid-East","E45": "Mid-West",
    "Y34": "South-East","W12": "Mid-East","V42": "Mid-West","A45": "Fingal","R32": "Midland","A67": "Mid-East","F42": "West","E53": "Mid-West",
    "K56": "Fingal","V14": "Mid-West","K34": "Fingal","P81": "South-West","F91": "Border","A83": "Mid-East","K67": "Fingal","E41": "Mid-West",
    "E34": "Mid-West","V92": "South-West","H54": "West","R35": "Midland","A82": "Border","X91": "South-East","F28": "West","Y35": "South-East",
    "A67": "Mid-East","P36": "South-West"
}

# Dictionary to convert month number to month name
month_to_word = {

    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"

}


df['date_listed'] = pd.to_datetime(df['date_listed'], format='mixed')

#Convert datetime to year and month as a word as that is how the RPPI dataset is formatted

df['year'] = df['date_listed'].dt.year
df['month'] = df['date_listed'].dt.month.map(month_to_word)
df['RPPI_time'] = df['year'].astype(str) + " " +  df['month']

df['newRegion'] = df['eircode'].str[:3].map(eircode_to_region)
df['RPPI_House_Type'] = df['newRegion'] + " - " + "houses"

print(df[['date_listed', 'year', 'month', 'RPPI_House_Type', 'RPPI_time']])

#Merge datasets on the newly created columns
merged_df = df.merge(df2, left_on=['RPPI_House_Type', 'RPPI_time'], right_on=['Type of Residential Property', 'Month'], how='left')

merged_df = merged_df.drop(columns=['Type of Residential Property', 'Month', 'UNIT', 'Statistic Label', 'newRegion', 'RPPI_House_Type', 'RPPI_time', 'year', 'month'])
merged_df = merged_df.rename(columns={'VALUE': 'RPPI'})

#save the dataset
merged_df.to_csv(os.getenv('NewHousePath'), index=False, encoding="utf-8")