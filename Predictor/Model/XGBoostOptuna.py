import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import os 
from dotenv import load_dotenv
import optuna
from optuna.samplers import TPESampler

# Load environment variables
load_dotenv()

# Load dataset
df = pd.read_csv(os.getenv("NewHousePath"))

# Convert date to Unix timestamp
df["date_listed"] = pd.to_datetime(df["date_listed"], errors="coerce").astype('int64') / 10**9

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

# Drop rows with any missing values after removing outliers
df.dropna(subset=features + [target], inplace=True)

X = df[features].values
y = df[target].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define cross-validation strategy
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Define custom scoring metrics
def mape_scorer(y_true, y_pred):
    y_true = np.expm1(y_true)
    y_pred = np.expm1(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def within_10pct_scorer(y_true, y_pred):
    y_true = np.expm1(y_true)
    y_pred = np.expm1(y_pred)
    within_10pct = np.abs((y_pred - y_true) / y_true) <= 0.10
    return np.mean(within_10pct) * 100

scoring = {
    'MAE': make_scorer(lambda y_true, y_pred: mean_absolute_error(np.expm1(y_true), np.expm1(y_pred))),
    'MSE': make_scorer(lambda y_true, y_pred: mean_squared_error(np.expm1(y_true), np.expm1(y_pred))),
    'RMSE': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(np.expm1(y_true), np.expm1(y_pred)))),
    'MAPE': make_scorer(mape_scorer),
    'R2': make_scorer(lambda y_true, y_pred: r2_score(np.expm1(y_true), np.expm1(y_pred))),
    'Within_10pct': make_scorer(within_10pct_scorer)
}

# Define Optuna objective function
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42
    }
    
    model = xgb.XGBRegressor(**params)
    
    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y, cv=kfold, scoring=scoring, verbose=0, return_train_score=False
    )
    
    # Return the average RMSE as the objective value
    return np.mean(cv_results['test_MAPE'])

# Run Optuna optimization
sampler = TPESampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=100)

# Print the best hyperparameters
print("Best Hyperparameters:", study.best_params)

# Train the final model with the best hyperparameters
best_params = study.best_params
final_model = xgb.XGBRegressor(**best_params, random_state=42)

# Perform cross-validation with the best model
final_cv_results = cross_validate(
    final_model, X, y, cv=kfold, scoring=scoring, verbose=10, return_train_score=False
)

# Calculate average metrics across folds
avg_mae = np.mean(final_cv_results['test_MAE'])
avg_mse = np.mean(final_cv_results['test_MSE'])
avg_rmse = np.mean(final_cv_results['test_RMSE'])
avg_mape = np.mean(final_cv_results['test_MAPE'])
avg_r2 = np.mean(final_cv_results['test_R2'])
avg_within_10pct = np.mean(final_cv_results['test_Within_10pct'])

# Print results
print(f"Average MAE: {avg_mae:.2f}")
print(f"Average MSE: {avg_mse:.2f}")
print(f"Average RMSE: {avg_rmse:.2f}")
print(f"Average MAPE: {avg_mape:.2f}%")
print(f"Average R² Score: {avg_r2:.4f}")
print(f"Average Percentage of predictions within 10% of actual value: {avg_within_10pct:.2f}%")