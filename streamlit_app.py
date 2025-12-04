import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))   # dict of LabelEncoders

st.title("HR Attrition Prediction (Manual Preprocessing)")

def preprocess_input(df):
    # Apply label encoders
    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])

    # Scale numeric columns
    num_cols = scaler.feature_names_in_
    df[num_cols] = scaler.transform(df[num_cols])

    return df

# Input UI
age = st.number_input("Age", 18, 60)
income = st.number_input("Monthly Income", 1000, 100000)
num_comp = st.number_input("Num Companies Worked", 0, 10)
years_company = st.number_input("Years at Company", 0, 40)
years_role = st.number_input("Years in Current Role", 0, 20)

overtime = st.selectbox("OverTime", ["Yes", "No"])
jobsat = st.slider("Job Satisfaction", 1, 4)
wlb = st.slider("Work Life Balance", 1, 4)
envsat = st.slider("Environment Satisfaction", 1, 4)
edu = st.slider("Education Level", 1, 5)

edu_field = st.selectbox("Education Field", ["Engineering", "Technical", "Human Resources", "Marketing"])
jobrole = st.selectbox("Job Role", ["Manufacturing Director", "Sales Executive", "Manager"])
dept = st.selectbox("Department", ["Operations", "R&D", "Sales"])
gender = st.selectbox("Gender", ["Male", "Female"])
marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

if st.button("Predict"):
    df = pd.DataFrame([{
        "Age": age,
        "MonthlyIncome": income,
        "NumCompaniesWorked": num_comp,
        "YearsAtCompany": years_company,
        "YearsInCurrentRole": years_role,
        "OverTime": overtime,
        "JobSatisfaction": jobsat,
        "WorkLifeBalance": wlb,
        "EnvironmentSatisfaction": envsat,
        "Education": edu,
        "EducationField": edu_field,
        "JobRole": jobrole,
        "Department": dept,
        "Gender": gender,
        "MaritalStatus": marital
    }])

    df_preprocessed = preprocess_input(df)

    pred = model.predict(df_preprocessed)[0]
    prob = model.predict_proba(df_preprocessed)[0][1]

    st.write("### Prediction:", "Employee Will Leave" if pred==1 else "Employee Will Stay")
    st.write("### Probability:", round(prob, 2))
