import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load dataset
df = pd.read_csv(os.getenv("NewHousePath"))

df["date_listed"] = df["date_listed"].str.replace(r"\.\d+$", "", regex=True)

# Convert date_listed to datetime
df["date_listed"] = pd.to_datetime(df["date_listed"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

# Debug: Check for NaT values after parsing
print("Rows with NaT in date_listed:", df["date_listed"].isna().sum())

# Extract the year and month from the date_listed column
df['year'] = df['date_listed'].dt.year
df['month'] = df['date_listed'].dt.month

df['date_listed'] = df['date_listed'].astype(int) / 10**9

# Debug: Print unique years and row counts
print("Unique years in the dataset:", df['year'].unique())
print("Number of rows per year:")
print(df['year'].value_counts().sort_index())

# Ensure valid years
valid_years = (df['year'] >= 2000) & (df['year'] <= 2100)
df = df[valid_years]

# Debug: Check rows for 2021 after filtering
print("Rows for 2021 after filtering:", len(df[df['year'] == 2021]))

df["price_per_sqm"] = df["sold_price"] / df["square_metres"]

# Remove outliers based on IQR for price_per_sqm, beds, and bathrooms
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df = remove_outliers(df, "price_per_sqm")
df = remove_outliers(df, "beds")
df = remove_outliers(df, "bathrooms")

# Debug: Check rows for 2021 after outlier removal
print("Rows for 2021 after outlier removal:", len(df[df['year'] == 2021]))

# Log-transform target to reduce skew
df["price_per_sqm"] = np.log1p(df["price_per_sqm"])

# One-hot encode BER
df = pd.get_dummies(df, columns=["ber"], prefix="ber")

# Map property type
house_to_type = {
    "House": "houses", "Detached House": "houses", "Semi-Detached House": "houses", 
    "Terraced House": "houses", "End of Terrace": "houses", "Bungalow": "houses", 
    "Townhouse": "houses", "Apartment": "apartments", "Duplex": "apartments", 
    "Studio": "apartments", "Unspecified": "houses"
}
df["property_category"] = df["property_type"].map(house_to_type)

# One-hot encode property category
df = pd.get_dummies(df, columns=["property_category"], prefix="prop")

# Select features
features = [
    "beds", "bathrooms", "square_metres", "date_listed", "region_id", "locality_id", 
    "latitude", "longitude", "nearest_secondary_distance", "nearest_mainstream_distance", 
    "nearest_special_distance", "avg_dist_to_10_shops", "RPPI", "Mean Sale Price", 
    "Median Sale Price", "region_AGEDEP22", "region_LONEPA22", "region_EDLOW_22", 
    "region_EDHIGH22", "region_HLPROF22", "region_LCLASS22", "region_UNEMPM22", 
    "region_UNEMPF22", "region_OWNOCC22", "region_PRRENT22", "region_LARENT22", 
    "region_PEROOM22", "region_Index22_ED_std_rel_wt", "region_Index22_ED_std_abs_wt", 
    "region_Index22_ED_rel_wt_cat"
]

# Add one-hot encoded columns to features
features.extend([col for col in df.columns if col.startswith("ber_") or col.startswith("prop_")])

target = "price_per_sqm"
print(features)

# Drop rows with any missing values after removing outliers
df.dropna(subset=features + [target], inplace=True)

# Debug: Check rows for 2021 after dropping NaNs
print("Rows for 2021 after dropping NaNs:", len(df[df['year'] == 2021]))

# Normalize features
scaler = StandardScaler()
scaler.fit(df[features])

# Define the XGBoost model
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1918,
    learning_rate=0.04532104920387173,
    max_depth=8,
    subsample=0.7752009360887773,
    colsample_bytree=0.6272489542893106,
    random_state=42,
    eval_metric="rmse"
)

# Initialize lists to store metrics
maes, mapes, mdapes, rmses, r2_scores, within_5, within_10, within_20 = [], [], [], [], [], [], [], []

# Define the months and years to iterate over
test_months = [(2022, month) for month in range(1, 13)] + [(2023, month) for month in range(1, 13)]

# Function to calculate the percentage of properties within a given error threshold
def calculate_within_threshold(actual, predicted, threshold):
    error_percentage = np.abs((actual - predicted) / actual) * 100
    within_threshold = np.sum(error_percentage <= threshold) / len(actual) * 100
    return within_threshold

# Iterate over the months for training and testing
for year, month in test_months:
    # Split data into training and testing sets
    if month == 1:
        # For January, train on all data up to December of the previous year
        train_df = df[df['year'] < year]
    else:
        # For other months, train on all data up to the previous month of the same year
        train_df = df[(df['year'] < year) | ((df['year'] == year) & (df['month'] < month))]
    
    test_df = df[(df['year'] == year) & (df['month'] == month)]
    
    # Debug: Print the months being used for training and testing
    print(f"Training data up to: {year-1 if month == 1 else year}-{month-1 if month > 1 else 12}-31, Testing month: {year}-{month}")
    
    # Skip if there's not enough data
    if len(train_df) < 10 or len(test_df) < 5:
        print(f"Skipping split: train={len(train_df)}, test={len(test_df)}")
        continue
    
    # Prepare features and target
    X_train = scaler.transform(train_df[features])
    y_train = train_df[target].values
    X_test = scaler.transform(test_df[features])
    y_test = test_df[target].values
    
    # Train the model
    xgb_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = xgb_model.predict(X_test)
    
    # Reverse log-transform for evaluation
    y_test_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(y_pred)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
    mdape = np.median(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    r2 = r2_score(y_test_actual, y_pred_actual)
    
    # Calculate the percentage of properties within 5%, 10%, and 20% error
    within_5.append(calculate_within_threshold(y_test_actual, y_pred_actual, 5))
    within_10.append(calculate_within_threshold(y_test_actual, y_pred_actual, 10))
    within_20.append(calculate_within_threshold(y_test_actual, y_pred_actual, 20))
    
    # Append metrics to lists
    maes.append(mae)
    mapes.append(mape)
    mdapes.append(mdape)
    rmses.append(rmse)
    r2_scores.append(r2)
    
    print(f"Month {year}-{month}: MAE={mae:.2f}, MAPE={mape:.2f}%, MdAPE={mdape:.2f}%, RMSE={rmse:.2f}, R²={r2:.4f}")
    print(f"  Properties within 5% error: {within_5[-1]:.2f}%")
    print(f"  Properties within 10% error: {within_10[-1]:.2f}%")
    print(f"  Properties within 20% error: {within_20[-1]:.2f}%")
    print("-----------------------------------------------------------")

# Calculate average metrics
avg_mae = np.mean(maes)
avg_mape = np.mean(mapes)
avg_mdape = np.mean(mdapes)
avg_rmse = np.mean(rmses)
avg_r2 = np.mean(r2_scores)
avg_within_5 = np.mean(within_5)
avg_within_10 = np.mean(within_10)
avg_within_20 = np.mean(within_20)

# Print average results
print("\nAverage Metrics Across All Splits:")
print(f"Average MAE: {avg_mae:.2f}")
print(f"Average MAPE: {avg_mape:.2f}%")
print(f"Average MdAPE: {avg_mdape:.2f}%")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average R² Score: {avg_r2:.4f}")
print(f"Average Properties within 5% error: {avg_within_5:.2f}%")
print(f"Average Properties within 10% error: {avg_within_10:.2f}%")
print(f"Average Properties within 20% error: {avg_within_20:.2f}%")