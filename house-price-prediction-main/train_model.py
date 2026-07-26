import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def main():
    print("=== Training House Price Prediction Model ===")
    
    csv_path = "housing data Book1.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    # 1. Load data
    data = pd.read_csv(csv_path)
    
    # 2. Filter to clean columns
    real_cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = data[[c for c in real_cols if c in data.columns]].copy()
    
    # 3. Handle missing values
    # The 'RM' column has 5 missing values. Let's fill them with the mean.
    data['RM'] = data['RM'].fillna(data['RM'].mean())
    
    # 4. Feature Engineering
    data['ROOMS_PER_AGE'] = data['RM'] / (data['AGE'] + 1)
    data['TAX_PTRATIO'] = data['TAX'] / (data['PTRATIO'] + 1)
    data['LSTAT_RM'] = data['LSTAT'] * data['RM']
    
    # Define features and target
    X = data.drop("MEDV", axis=1)
    y = data["MEDV"]
    
    # 5. Extract metadata for the web app (ranges, averages)
    defaults = X.mean().to_dict()
    # Replace any nan with 0 just in case
    defaults = {k: (0.0 if np.isnan(v) else float(v)) for k, v in defaults.items()}
    
    min_max = {}
    for col in X.columns:
        min_max[col] = {
            "min": float(X[col].min()),
            "max": float(X[col].max())
        }
        
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Train RandomForestRegressor Model
    print("Training RandomForestRegressor model...")
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    
    # 7. Evaluate model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n--- Model Performance Summary ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # 8. Save the package containing the model and metadata
    package_path = "house_price_model.joblib"
    package = {
        "model": model,
        "features": list(X.columns),
        "defaults": defaults,
        "min_max": min_max
    }
    
    joblib.dump(package, package_path)
    print(f"\nModel and metadata package saved successfully to: '{package_path}'")

if __name__ == "__main__":
    main()
