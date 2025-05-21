from flask import Flask, request, jsonify
import mlflow.pyfunc
import mlflow
import pandas as pd
import pickle
import os

# === Config ===
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_URI = "models:/churn_prediction_experiment/1"
ENCODER_PATH = "/opt/airflow/dags/label_encoders.pkl"  # adjust as needed in your Docker volume

# === Init app ===
app = Flask(__name__)
model = None
encoders = {}

# === Load model and encoders ===
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model from MLflow: {e}")

try:
    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)
    print("✅ Label encoders loaded")
except Exception as e:
    print(f"❌ Failed to load encoders: {e}")

# === Routes ===
@app.route("/")
def home():
    return "✅ Churn Prediction API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # Encode only categorical columns that were encoded during training
        for col, le in encoders.items():
            if col in input_df.columns:
                val = input_df[col].values[0]
                if val not in le.classes_:
                    return jsonify({"error": f"Unseen label '{val}' in column '{col}'"}), 400
                input_df[col] = le.transform([val])

        # Numeric columns like TotalCharges are left untouched

        prediction = model.predict(input_df)
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Run App ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055)
