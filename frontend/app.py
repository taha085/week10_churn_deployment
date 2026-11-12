"""Streamlit client for the Week 9 FastAPI prediction service."""
import os
import requests
import streamlit as st

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📊", layout="centered")
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.title("Telco Customer Churn Predictor")
st.caption("A demonstration interface powered by the Week 9 FastAPI model service.")
st.info("Enter a customer's service and account details. The UI sends the form to the API; it does not load the model.")

with st.form("customer_form"):
    st.subheader("Customer profile")
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
    with c2:
        device = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        tech = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly = st.number_input("Monthly charges", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
        total = st.number_input("Total charges", min_value=0.0, max_value=10000.0, value=840.0, step=1.0)
    submitted = st.form_submit_button("Predict churn", type="primary", use_container_width=True)

if submitted:
    payload = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multiple, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": device, "TechSupport": tech, "StreamingTV": tv,
        "StreamingMovies": movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }
    try:
        with st.spinner("Requesting prediction from FastAPI..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=20)
        if response.status_code == 200:
            result = response.json()
            probability = result["churn_probability"]
            st.subheader("Prediction result")
            st.metric("Churn probability", f"{probability:.1%}")
            st.progress(min(max(probability, 0.0), 1.0))
            if result["prediction"] == "Yes":
                st.error("Predicted outcome: Likely to churn")
            else:
                st.success("Predicted outcome: Unlikely to churn")
            st.caption(f"Decision threshold: {result['threshold']:.4f}")
            st.caption("This is a model estimate, not a guarantee of customer behavior.")
        else:
            st.error(f"API returned {response.status_code}: {response.text[:500]}")
    except requests.RequestException as exc:
        st.error(f"Could not reach the prediction API at {API_URL}. Check that the backend is running. Details: {exc}")

with st.expander("API connection details"):
    st.code(f"{API_URL}/predict")
    st.write("Configure the backend address with the API_URL environment variable.")
