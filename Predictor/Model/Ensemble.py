import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_squared_log_error, mean_absolute_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load and preprocess data
df = pd.read_csv(os.getenv("NewHousePath"))
df["date_listed"] = pd.to_datetime(df["date_listed"], errors="coerce")
df["year_listed"] = df["date_listed"].dt.year
df["month_listed"] = df["date_listed"].dt.month
df["day_listed"] = df["date_listed"].dt.day
df["date_listed"] = df["date_listed"].astype('int64') / 10**9
df["price_per_sqm"] = df["sold_price"] / df["square_metres"]

# Remove outliers
numeric_cols = df[['sold_price', 'beds', 'bathrooms', 'square_metres', 'price_per_sqm']]
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

df.dropna(inplace=True)

house_to_type = {
    "House" : "houses", "Detached House" : "houses", "Semi-Detached House" : "houses", "Terraced House" : "houses", "Detached House" : "houses", "End of Terrace" : "houses", "Bungalow" : "houses", "Townhouse" : "houses",
    "Apartment" : "apartments", "Duplex" : "apartments", "Studio" : "apartments" ,
    "Unspecified" : "houses"
}
df["property_category"] = df["property_type"].map(house_to_type)
df = pd.get_dummies(df, columns=["property_category"], prefix="prop")

# Define base features
base_features = [
    "beds", "bathrooms", "square_metres", "date_listed", "year_listed", "month_listed", "day_listed", "region_id", "locality_id", "latitude", "longitude",
    "nearest_secondary_distance", "nearest_mainstream_distance", "nearest_special_distance", "avg_dist_to_10_shops",
    "RPPI", "Mean Sale Price", "Median Sale Price", "region_AGEDEP22", "region_LONEPA22", "region_EDLOW_22",
    "region_EDHIGH22", "region_HLPROF22", "region_LCLASS22", "region_UNEMPM22", "region_UNEMPF22", "region_OWNOCC22",
    "region_PRRENT22", "region_LARENT22", "region_PEROOM22", "region_Index22_ED_std_rel_wt",
    "region_Index22_ED_std_abs_wt", "region_Index22_ED_rel_wt_cat"
]
base_features.extend([col for col in df.columns if col.startswith("ber_") or col.startswith("prop_")])
X_nn = df[base_features]
y_final = df['price_per_sqm']

# Split data into training and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X_nn, y_final, test_size=0.2, random_state=42)

# Initialize KFold
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store evaluation metrics for the test set
test_mae_scores, test_mape_scores, test_r2_scores, test_rmse_scores, test_rmsle_scores, test_adjusted_r2_scores, test_within_5_percent_scores, test_within_10_percent_scores, test_within_20_percent_scores = [], [], [], [], [], [], [], [], []

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the neural network model
class PricePredictionNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super(PricePredictionNN, self).__init__()
        layers = []
        previous_dim = input_dim
        for hidden_layer in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_layer))
            layers.append(nn.SELU())
            layers.append(nn.Dropout(dropout_rate))
            previous_dim = hidden_layer
        layers.append(nn.Linear(previous_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Training function for the neural network
def train_model(model, criterion, optimizer, train_loader, val_loader=None, epochs=200, patience=20):
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_loader.dataset)

        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    predictions = model(X_batch).squeeze()
                    loss = criterion(predictions, y_batch)
                    val_loss += loss.item() * len(X_batch)
            val_loss /= len(val_loader.dataset)
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
        else:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}")

    model.load_state_dict(torch.load('best_model.pth'))
    return model

# Add these lists to store median errors for each fold
test_median_errors, test_median_percentage_errors = [], []

# Perform K-Fold Cross-Validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
    print(f"Fold {fold + 1}/{n_splits}")
    
    # Split data into training and validation sets
    X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
    y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
    
    # Standardize features
    scaler_nn = StandardScaler()
    X_train_scaled = scaler_nn.fit_transform(X_train)
    X_val_scaled = scaler_nn.transform(X_val)
    X_test_scaled = scaler_nn.transform(X_test)  # Standardize the test set using the same scaler
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Define and train the neural network
    input_dim_nn = X_train_scaled.shape[1]
    hidden_layers = [445, 393]
    dropout_rate = 0.1368395
    learning_rate = 0.00564375
    
    model = PricePredictionNN(input_dim_nn, hidden_layers, dropout_rate).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    X_val_tensor = X_val_tensor.to(device)
    y_val_tensor = y_val_tensor.to(device)
    
    trained_model = train_model(model, criterion, optimizer, train_loader, val_loader, epochs=200, patience=20)
    
    # Evaluate the neural network on the validation set
    model.eval()
    with torch.no_grad():
        nn_val_pred = trained_model(X_val_tensor).detach().cpu().numpy().squeeze()
        nn_test_pred = trained_model(X_test_tensor.to(device)).detach().cpu().numpy().squeeze()
    
    # Train RandomForest
    rf_reg = RandomForestRegressor(n_estimators=1164, random_state=0)
    rf_reg.fit(X_train_scaled, y_train)
    rf_val_pred = rf_reg.predict(X_val_scaled)
    rf_test_pred = rf_reg.predict(X_test_scaled)
    
    # Train XGBoost
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=2278, learning_rate=0.012492184015888655, max_depth=12,
        subsample=0.7103181018871908, colsample_bytree=0.773830656506188, random_state=42, eval_metric="rmse"
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_val_pred = xgb_model.predict(X_val_scaled)
    xgb_test_pred = xgb_model.predict(X_test_scaled)
    
    # Stack predictions
    stacked_val_features = np.column_stack((rf_val_pred, nn_val_pred, xgb_val_pred))
    stacked_test_features = np.column_stack((rf_test_pred, nn_test_pred, xgb_test_pred))
    
    # Train meta-model (XGBoost)
    meta_model = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=2124, learning_rate=0.00645798338666791, max_depth=3,
        subsample=0.5285079977377378, colsample_bytree=0.925719582831838, random_state=42, eval_metric="rmse"
    )
    meta_model.fit(stacked_val_features, y_val)
    
    # Final predictions on the test set
    final_test_pred = meta_model.predict(stacked_test_features)
    
    # Calculate absolute errors and percentage errors
    absolute_errors = np.abs(final_test_pred - y_test)
    percentage_errors = np.abs((final_test_pred - y_test) / y_test) * 100
    
    # Calculate median error and median percentage error
    median_error = np.median(absolute_errors)
    median_percentage_error = np.median(percentage_errors)
    
    # Store median errors
    test_median_errors.append(median_error)
    test_median_percentage_errors.append(median_percentage_error)
    
    # Evaluate on the test set
    mae = mean_absolute_error(y_test, final_test_pred)
    mape = mean_absolute_percentage_error(y_test, final_test_pred) * 100
    r2 = r2_score(y_test, final_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, final_test_pred))
    y_test_non_negative = np.maximum(y_test, 0)
    final_test_pred_non_negative = np.maximum(final_test_pred, 0)
    rmsle = np.sqrt(mean_squared_log_error(y_test_non_negative, final_test_pred_non_negative))
    n = len(y_test)
    p = X_test_scaled.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    within_5_percent = np.mean(np.abs((final_test_pred - y_test) / y_test) <= 0.05) * 100
    within_10_percent = np.mean(np.abs((final_test_pred - y_test) / y_test) <= 0.1) * 100
    within_20_percent = np.mean(np.abs((final_test_pred - y_test) / y_test) <= 0.2) * 100
    
    # Store test set metrics
    test_mae_scores.append(mae)
    test_mape_scores.append(mape)
    test_r2_scores.append(r2)
    test_rmse_scores.append(rmse)
    test_rmsle_scores.append(rmsle)
    test_adjusted_r2_scores.append(adjusted_r2)
    test_within_5_percent_scores.append(within_5_percent)
    test_within_10_percent_scores.append(within_10_percent)
    test_within_20_percent_scores.append(within_20_percent)

# Print average metrics across folds for the test set
print("Test Set Results (Averaged Across Folds):")
print(f"Average MAE: {np.mean(test_mae_scores):.2f}")
print(f"Average MAPE: {np.mean(test_mape_scores):.2f}%")
print(f"Average R²: {np.mean(test_r2_scores):.4f}")
print(f"Average RMSE: {np.mean(test_rmse_scores):.2f}")
print(f"Average RMSLE: {np.mean(test_rmsle_scores):.2f}")
print(f"Average Adjusted R²: {np.mean(test_adjusted_r2_scores):.2f}")
print(f"Average Percentage of Predictions Within 5% of Actual Price: {np.mean(test_within_5_percent_scores):.2f}%")
print(f"Average Percentage of Predictions Within 10% of Actual Price: {np.mean(test_within_10_percent_scores):.2f}%")
print(f"Average Percentage of Predictions Within 20% of Actual Price: {np.mean(test_within_20_percent_scores):.2f}%")
print(f"Average Median Error: {np.mean(test_median_errors):.2f}")
print(f"Average Median Percentage Error: {np.mean(test_median_percentage_errors):.2f}%")