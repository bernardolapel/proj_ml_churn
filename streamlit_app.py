import streamlit as st
import requests
from matplotlib import pyplot as plt
import os

st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.title("📊Churn Prediction App")

# Collect input features
gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No Phone Service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
online_backup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])
device_protection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
tech_support = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
streaming_tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0, max_value=10000, value=4000 )

API_URL = "http://churn-api:5055/predict"  # or localhost:5055 if running locally
# Support both local and Docker deployments
# API_URL = os.getenv("API_URL", "http://localhost:5055/predict")

# When the user clicks 'Predict'
if st.button("Predict"):
    input_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
        # Add other expected features here
    }

    with st.spinner("Making prediction..."):
        try:
            response = requests.post(API_URL, json=input_data)
            response.raise_for_status()
            result = response.json()

            if "prediction" in result:
                pred_label = "Churn" if result["prediction"] == 1 else "No Churn"
                st.success(f"Prediction: {pred_label}")

                # Optional: Show visual confidence score if your API returns it later
                fig, ax = plt.subplots()
                ax.bar(["No Churn", "Churn"], [1 - result["prediction"], result["prediction"]], color=["green", "red"])
                ax.set_ylabel("Confidence")
                ax.set_ylim(0, 1)
                st.pyplot(fig)
            else:
                st.error(f"❌ Unexpected response format:\n{result}")

        except requests.exceptions.RequestException as e:
            st.error(f"🚫 API request failed: {e}")
        except ValueError:
            st.error("🚫 Response is not valid JSON.")

