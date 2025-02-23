from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from irrigation import IrrigationSystem  # Import your ML model class

app = Flask(__name__)

# Load the dataset
DATASET_PATH = "data/irrigation_strategy_with_soil_type.csv"  # Update if needed
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("Dataset file not found! Make sure it is uploaded.")

# Initialize model
irrigation_model = IrrigationSystem(DATASET_PATH)

# Extract unique options from dataset
df = pd.read_csv(DATASET_PATH)
AVAILABLE_CROPS = sorted(df["crop"].unique().tolist())
AVAILABLE_SEASONS = sorted(df["season"].unique().tolist())
AVAILABLE_ALTITUDES = sorted(df["altitude"].unique().tolist())
AVAILABLE_SOIL_TYPES = sorted(df["soil_type"].unique().tolist())

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/options', methods=['GET'])
def options():
    """Fetch unique values from dataset and send to frontend"""
    return jsonify({
        "available_crops": AVAILABLE_CROPS,
        "available_seasons": AVAILABLE_SEASONS,
        "available_altitudes": AVAILABLE_ALTITUDES,
        "available_soil_types": AVAILABLE_SOIL_TYPES
    })

@app.route('/predict', methods=['POST'])
def predict():
    """API to predict irrigation strategy and water requirement."""
    try:
        # Get JSON data from the request
        data = request.json

        # Convert JSON data to DataFrame
        df = pd.DataFrame([data])

        # Preprocess the input just like training data
        for col, encoder in irrigation_model.label_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])

        # Make predictions
        predicted_strategy = irrigation_model.classifier.predict(df)[0]
        predicted_water_req = irrigation_model.regressor.predict(df)[0]

        # Convert predictions back to readable format
        strategy_label = irrigation_model.label_encoder_strategy.inverse_transform([predicted_strategy])[0]

        return jsonify({
            "predicted_strategy": strategy_label,
            "predicted_water_requirement": round(predicted_water_req, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
