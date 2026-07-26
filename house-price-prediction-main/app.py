import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model and metadata package
package_path = "house_price_model.joblib"
if os.path.exists(package_path):
    model_package = joblib.load(package_path)
    model = model_package["model"]
    feature_list = model_package["features"]
    default_vals = model_package["defaults"]
    min_max_vals = model_package["min_max"]
    print("Model and metadata loaded successfully.")
else:
    model = None
    feature_list = []
    default_vals = {}
    min_max_vals = {}
    print("WARNING: model package not found. Please train the model first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/metadata', methods=['GET'])
def get_metadata():
    """Return model defaults and min/max ranges for frontend controls."""
    return jsonify({
        "defaults": default_vals,
        "min_max": min_max_vals,
        "success": True
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"success": False, "error": "Model is not loaded."}), 500
        
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No input data provided."}), 400
            
        # Get raw 13 features with defaults if missing
        raw_features = {}
        for feature in ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']:
            raw_features[feature] = float(data.get(feature, default_vals.get(feature, 0.0)))
            
        # Calculate engineered features
        rm = raw_features['RM']
        age = raw_features['AGE']
        tax = raw_features['TAX']
        ptratio = raw_features['PTRATIO']
        lstat = raw_features['LSTAT']
        
        rooms_per_age = rm / (age + 1.0)
        tax_ptratio = tax / (ptratio + 1.0)
        lstat_rm = lstat * rm
        
        # Combine all features into the dictionary
        all_features = {**raw_features}
        all_features['ROOMS_PER_AGE'] = rooms_per_age
        all_features['TAX_PTRATIO'] = tax_ptratio
        all_features['LSTAT_RM'] = lstat_rm
        
        # Reconstruct DataFrame with exact column ordering
        df = pd.DataFrame([all_features])[feature_list]
        
        # Make prediction (returns value in $1000s)
        prediction_val = float(model.predict(df)[0])
        
        # Convert to actual dollar value (e.g. 24.5 -> $24,500)
        price_in_dollars = prediction_val * 1000.0
        
        return jsonify({
            "success": True,
            "prediction": round(price_in_dollars, 2),
            "features_used": all_features
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run House Price Prediction App")
    parser.add_argument('--prod', action='store_true', help="Run in production mode using Waitress WSGI server")
    parser.add_argument('--port', type=int, default=5000, help="Port to run the server on")
    args = parser.parse_args()

    if args.prod:
        try:
            from waitress import serve
            print(f"Starting production WSGI server (Waitress) on http://0.0.0.0:{args.port}...")
            serve(app, host='0.0.0.0', port=args.port)
        except ImportError:
            print("Error: 'waitress' library is not installed. Please install it using: pip install waitress")
    else:
        # Running in development debug mode on port 5000
        print(f"Starting development server on http://127.0.0.1:{args.port}...")
        app.run(debug=True, host='0.0.0.0', port=args.port)
