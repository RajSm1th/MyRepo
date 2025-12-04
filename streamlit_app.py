import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------------
# LOAD MODEL + PREPROCESSORS
# ---------------------------
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "encoders.pkl"

model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))
encoders = pickle.load(open(ENCODER_PATH, "rb"))   # dictionary of label encoders

st.title("🔮 HR Attrition Prediction App")
st.write("Enter employee details below to predict attrition probability.")


# ---------------------------
# STREAMLIT FORM UI
# ---------------------------
with st.form("prediction_form"):
    
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", 18, 65, 30)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Education = st.selectbox("Education", ["Basic", "Technical Degree", "Bachelor", "Master", "PhD"])
        MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        JobRole = st.selectbox("Job Role", ["Laboratory Technician","Sales Executive","Research Scientist","Manager"])

    with col2:
        MonthlyIncome = st.number_input("Monthly Income", 1000, 50000, 5000)
        TotalWorkingYears = st.number_input("Total Working Years", 0, 40, 5)
        YearsAtCompany = st.number_input("Years at Company", 0, 40, 3)
        OverTime = st.selectbox("OverTime", ["Yes", "No"])
        JobSatisfaction = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
    
    submitted = st.form_submit_button("Predict Attrition")


# ---------------------------
# MAKE PREDICTION
# ---------------------------
if submitted:

    input_dict = {
        "Age": Age,
        "Gender": Gender,
        "Education": Education,
        "MaritalStatus": MaritalStatus,
        "JobRole": JobRole,
        "MonthlyIncome": MonthlyIncome,
        "TotalWorkingYears": TotalWorkingYears,
        "YearsAtCompany": YearsAtCompany,
        "OverTime": OverTime,
        "JobSatisfaction": JobSatisfaction
    }

    df_input = pd.DataFrame([input_dict])


    # ---------------------------
    # LABEL ENCODING
    # ---------------------------
    for col, encoder in encoders.items():
        if col in df_input.columns:
            df_input[col] = encoder.transform(df_input[col])


    # ---------------------------
    # SCALING
    # ---------------------------
    df_scaled = scaler.transform(df_input)


    # ---------------------------
    # PREDICTION
    # ---------------------------
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]


    # ---------------------------
    # OUTPUT
    # ---------------------------
    st.subheader("📌 Prediction Result:")
    if pred == 1:
        st.error(f"Employee is LIKELY to leave. Probability = {prob:.2f}")
    else:
        st.success(f"Employee is NOT likely to leave. Probability = {prob:.2f}")
