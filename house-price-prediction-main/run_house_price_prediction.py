import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

def main():
    print("=== House Price Prediction ===")
    
    # 1. Load Dataset
    csv_path = "housing data Book1.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    print(f"Loading dataset from: {csv_path}")
    data = pd.read_csv(csv_path)
    
    # 2. Get Data Information
    print("\n--- Initial Data Info ---")
    data.info()
    
    print("\n--- Data Description ---")
    print(data.describe())
    
    print("\n--- Missing Values Count ---")
    print(data.isnull().sum())
    
    # 3. Data Preprocessing
    print("\n--- Filling Numeric Nulls with Mean ---")
    data.fillna(data.select_dtypes(include='number').mean(), inplace=True)
    
    # Drop duplicates
    print("\n--- Dropping Duplicates ---")
    initial_shape = data.shape
    data.drop_duplicates(inplace=True)
    print(f"Shape changed from {initial_shape} to {data.shape}")
    
    # One-hot encoding
    print("\n--- Applying One-Hot Encoding ---")
    data = pd.get_dummies(data)
    print(f"Shape after encoding: {data.shape}")
    
    # 4. Feature Target Split
    if 'MEDV' not in data.columns:
        print("Error: Target column 'MEDV' not found in dataset columns.")
        return
        
    x = data.drop("MEDV", axis=1)
    y = data["MEDV"]
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Fill remaining NaNs if any in train/test features
    x_train = x_train.fillna(x_train.mean())
    x_test = x_test.fillna(x_test.mean())
    
    # Align indices
    x_train = x_train.loc[y_train.index]
    y_train = y_train.loc[x_train.index]
    
    print("\n--- Data Shapes ---")
    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    
    # Drop all-NaN columns from data (retained from notebook, though it doesn't affect training)
    data = data.dropna(axis=1, how='all')
    
    # Impute missing values
    print("\n--- Imputing Missing Values ---")
    imputer = SimpleImputer(strategy='mean')
    x_train_imputed = imputer.fit_transform(x_train)
    x_test_imputed = imputer.transform(x_test)
    
    # 5. Model Training
    print("\n--- Training RandomForestRegressor Model ---")
    model = RandomForestRegressor(random_state=42)
    model.fit(x_train_imputed, y_train)
    
    # 6. Evaluation
    predictions = model.predict(x_test_imputed)
    
    print("\n--- Sample Predictions ---")
    print("Predicted:", predictions[:5])
    print("Actual:", y_test[:5].values)
    
    print("\n--- Evaluation Metrics ---")
    print('MAE:', mean_absolute_error(y_test, predictions))
    print('MSE:', mean_squared_error(y_test, predictions))
    print('R2 Score:', r2_score(y_test, predictions))
    
    # 7. Visualization
    print("\n--- Saving Scatter Plot ---")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.6, edgecolors='k')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices (Random Forest)")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plot_filename = "actual_vs_predicted.png"
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Plot saved successfully as '{plot_filename}'")
    
    print("\n=== Execution Completed ===")

if __name__ == "__main__":
    main()
