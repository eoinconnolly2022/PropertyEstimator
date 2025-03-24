import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import os 
from dotenv import load_dotenv
import joblib
load_dotenv()

# Load dataset
df = pd.read_csv(os.getenv("NewHousePath"))

df["date_listed"] = df["date_listed"].str.replace(r"\.\d+$", "", regex=True)

# Convert date_listed to datetime
df["date_listed"] = pd.to_datetime(df["date_listed"], format="%Y-%m-%d %H:%M:%S", errors="coerce").astype('int64') / 10**9

# Create target variable (price per sqm)
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

# Print number of rows after preprocessing
print(f"Number of rows after removing outliers and dropping NaNs: {len(df)}")

# Log-transform target to reduce skew
df["price_per_sqm"] = np.log1p(df["price_per_sqm"])

# One-hot encode BER
df = pd.get_dummies(df, columns=["ber"], prefix="ber")

# Map property type
house_to_type = {
    "House" : "houses", "Detached House" : "houses", "Semi-Detached House" : "houses", "Terraced House" : "houses", "Detached House" : "houses", "End of Terrace" : "houses", "Bungalow" : "houses", "Townhouse" : "houses",
    "Apartment" : "apartments", "Duplex" : "apartments", "Studio" : "apartments" ,
    "Unspecified" : "houses"
}
df["property_category"] = df["property_type"].map(house_to_type)

# One-hot encode property category
df = pd.get_dummies(df, columns=["property_category"], prefix="prop")

# Select features
features = [
    "beds", "bathrooms", "square_metres", "date_listed", "region_id", "locality_id", "latitude", "longitude",
    "nearest_secondary_distance", "nearest_mainstream_distance", "nearest_special_distance", "avg_dist_to_10_shops",
    "RPPI", "Mean Sale Price", "Median Sale Price", "region_AGEDEP22", "region_LONEPA22", "region_EDLOW_22",
    "region_EDHIGH22", "region_HLPROF22", "region_LCLASS22", "region_UNEMPM22", "region_UNEMPF22", "region_OWNOCC22",
    "region_PRRENT22", "region_LARENT22", "region_PEROOM22", "region_Index22_ED_std_rel_wt",
    "region_Index22_ED_std_abs_wt", "region_Index22_ED_rel_wt_cat"
]

# Add one-hot encoded columns to features
features.extend([col for col in df.columns if col.startswith("ber_") or col.startswith("prop_")])

target = "price_per_sqm"
print(features)
# Drop rows with any missing values after removing outliers
df.dropna(subset=features + [target], inplace=True)

X = df[features].values
y = df[target].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

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

# Define cross-validation strategy
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Define custom scoring metrics
def mape_scorer(y_true, y_pred):
    y_true = np.expm1(y_true)
    y_pred = np.expm1(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mdape_scorer(y_true, y_pred):
    y_true = np.expm1(y_true)
    y_pred = np.expm1(y_pred)
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100

def within_5pct_scorer(y_true, y_pred):
    y_true = np.expm1(y_true)
    y_pred = np.expm1(y_pred)
    within_5pct = np.abs((y_pred - y_true) / y_true) <= 0.05
    return np.mean(within_5pct) * 100

def within_10pct_scorer(y_true, y_pred):
    y_true = np.expm1(y_true)
    y_pred = np.expm1(y_pred)
    within_10pct = np.abs((y_pred - y_true) / y_true) <= 0.10
    return np.mean(within_10pct) * 100

def within_20pct_scorer(y_true, y_pred):
    y_true = np.expm1(y_true)
    y_pred = np.expm1(y_pred)
    within_20pct = np.abs((y_pred - y_true) / y_true) <= 0.20
    return np.mean(within_20pct) * 100

scoring = {
    'MAE': make_scorer(lambda y_true, y_pred: mean_absolute_error(np.expm1(y_true), np.expm1(y_pred))),
    'MSE': make_scorer(lambda y_true, y_pred: mean_squared_error(np.expm1(y_true), np.expm1(y_pred))),
    'RMSE': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(np.expm1(y_true)), np.expm1(y_pred))),
    'MAPE': make_scorer(mape_scorer),
    'MdAPE': make_scorer(mdape_scorer),
    'R2': make_scorer(lambda y_true, y_pred: r2_score(np.expm1(y_true), np.expm1(y_pred))),
    'Within_5pct': make_scorer(within_5pct_scorer),
    'Within_10pct': make_scorer(within_10pct_scorer),
    'Within_20pct': make_scorer(within_20pct_scorer)
}

# Perform cross-validation
cv_results = cross_validate(
    xgb_model, X, y, cv=kfold, scoring=scoring, verbose=10, return_train_score=False
)

# Calculate average metrics across folds
avg_mae = np.mean(cv_results['test_MAE'])
avg_mse = np.mean(cv_results['test_MSE'])
avg_rmse = np.mean(cv_results['test_RMSE'])
avg_mape = np.mean(cv_results['test_MAPE'])
avg_mdape = np.mean(cv_results['test_MdAPE'])
avg_r2 = np.mean(cv_results['test_R2'])
avg_within_5pct = np.mean(cv_results['test_Within_5pct'])
avg_within_10pct = np.mean(cv_results['test_Within_10pct'])
avg_within_20pct = np.mean(cv_results['test_Within_20pct'])

# Print results
print(f"Average MAE: {avg_mae:.2f}")
print(f"Average MSE: {avg_mse:.2f}")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAPE: {avg_mape:.2f}%")
print(f"Average MdAPE: {avg_mdape:.2f}%")
print(f"Average R² Score: {avg_r2:.4f}")
print(f"Average Percentage of predictions within 5% of actual value: {avg_within_5pct:.2f}%")
print(f"Average Percentage of predictions within 10% of actual value: {avg_within_10pct:.2f}%")
print(f"Average Percentage of predictions within 20% of actual value: {avg_within_20pct:.2f}%")

xgb_model.fit(X, y)


# Save the XGBoost model
joblib.dump(xgb_model, "xgboost_model.pkl")

# Save the StandardScaler
joblib.dump(scaler, "standard_scaler.pkl")
