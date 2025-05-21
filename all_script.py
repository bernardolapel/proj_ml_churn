# preprocess.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and clean data
df = pd.read_csv("Telco-Customer-Churn.csv")
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical features
categorical_cols = df.select_dtypes(include=['object']).columns.drop('customerID')
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Prepare features and target
X = df.drop(["customerID", "Churn"], axis=1)
y = df["Churn"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save datasets
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("✅ Preprocessing complete.")


# train.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load preprocessed data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

# Train model
model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# MLflow logging
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("churn_prediction_experiment")

with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="churn_prediction_experiment"
    )

print("✅ Model training and logging complete.")


# app.py
from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
import pickle

# Load model
model_uri = "models:/churn_prediction_experiment/1"
model = mlflow.pyfunc.load_model(model_uri)

# Load encoders
with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Churn Prediction API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # Encode categorical columns
        for col, le in encoders.items():
            if col in input_df:
                input_df[col] = le.transform([input_df[col].values[0]])

        prediction = model.predict(input_df)
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# Makefile for ML Churn Prediction Project

.PHONY: up build down logs train

# Build all services
build:
	docker-compose build

# Start all services
up:
	docker-compose up --build

# Stop and remove containers
stop:
	docker-compose down

# View logs from all services
logs:
	docker-compose logs -f

# Trigger training DAG manually (requires airflow CLI inside container)
train:
	docker exec -it airflow-airflow-1 airflow dags trigger ml_churn_training_pipeline

# View MLflow in browser
mlflow:
	@echo "Visit: http://localhost:5001"

# View Streamlit in browser
streamlit:
	@echo "Visit: http://localhost:8501"
