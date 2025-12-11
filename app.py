import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Construction AI Predictor", page_icon="🏗️")

st.title("🏗️ AI-based Construction Project Prediction System")
st.write("""
This web app predicts:
- **Final Project Cost (SAR)**
- **Probability of Delay (%)**
Based on realistic simulated company data.
""")

# Project Types
project_types = [
    "Residential Building",
    "Non-Residential Building",
    "Electrical Works",
    "Network & Communication",
    "Finishing & Tiling",
    "Renovation",
    "Digital Screen Installation"
]

p_type = st.selectbox("Project Type", project_types)

size = st.number_input("Project Size (m²)", min_value=1, value=100)
workers = st.number_input("Number of Workers", min_value=1, value=5)
budget = st.number_input("Estimated Budget (SAR)", min_value=1000, value=50000)
duration = st.number_input("Expected Duration (months)", min_value=1, value=2)

# Prepare input row
def prepare_input():
    row = {
        "Project_Size": size,
        "Num_Workers": workers,
        "Budget": budget,
        "Duration": duration,
        "Cost_Pressure": budget / size,
        "Worker_Density": workers / size,
    }
    for t in project_types:
        row[f"Project_Type_{t}"] = 1 if p_type == t else 0
    return pd.DataFrame([row])

input_data = prepare_input()

# Load models
reg_model = joblib.load("regression_model.pkl")
clf_model = joblib.load("classification_model.pkl")

# Predictions
predicted_cost = reg_model.predict(input_data)[0]
delay_probability = clf_model.predict_proba(input_data)[0][1] * 100

st.subheader("🔍 Prediction Results")

st.metric("Predicted Final Cost (SAR)", f"{predicted_cost:,.0f}")
st.metric("Delay Probability (%)", f"{delay_probability:.1f}%")
