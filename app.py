import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Construction AI Predictor", page_icon="🏗️", layout="centered")

st.title("🏗️ AI-based Construction Project Prediction System")
st.write("""
This system predicts:
- **Final Project Cost (SAR)**
- **Probability of Delay (%)**

Based on realistic company-style simulated construction data.
It also provides **smart recommendations** to support decision-making.
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

# User Inputs
p_type = st.selectbox("Project Type", project_types)
size = st.number_input("Project Size (m²)", min_value=1, value=150)
workers = st.number_input("Number of Workers", min_value=1, value=10)
budget = st.number_input("Estimated Budget (SAR)", min_value=1000, value=100000)
duration = st.number_input("Expected Duration (months)", min_value=1, value=6)

# Load models
reg_model = joblib.load("regression_model.pkl")
clf_model = joblib.load("classification_model.pkl")

# Build input row EXACTLY like training dataset structure
def prepare_input():
    row = {
        "Project_Size": size,
        "Num_Workers": workers,
        "Budget": budget,
        "Duration": duration,
        "Cost_Pressure": budget / size,
        "Worker_Density": workers / size,
    }

    # one-hot encoding for project type
    for t in project_types:
        row[f"Project_Type_{t}"] = 1 if p_type == t else 0

    return pd.DataFrame([row])

input_data = prepare_input()

# ---- Predictions ----
predicted_cost = reg_model.predict(input_data)[0]
delay_probability = clf_model.predict_proba(input_data)[0][1] * 100

st.subheader("🔍 Prediction Results")

st.metric("Predicted Final Cost (SAR)", f"{predicted_cost:,.0f}")
st.metric("Delay Probability (%)", f"{delay_probability:.1f}%")

# ---- Smart Recommendations ----
st.subheader("🛠️ Smart Recommendations")

if delay_probability > 70:
    st.error("🔴 **High Delay Risk**")
    st.write("""
    - Increase workforce to accelerate progress  
    - Extend planned duration to avoid penalties  
    - Increase budget buffer by +5%  
    - Monitor material delivery schedules  
    """)
elif delay_probability > 40:
    st.warning("🟠 **Moderate Delay Risk**")
    st.write("""
    - Review contractor availability  
    - Ensure material suppliers are consistent  
    - Add 2–3 additional workers if possible  
    """)
else:
    st.success("🟢 **Low Delay Risk**")
    st.write("""
    - Current plan is stable  
    - Maintain workforce levels  
    - Continue regular progress monitoring  
    """)

# ---- Extra insights (new feature!) ----
st.subheader("📊 Insights")

st.write(f"**Cost Pressure:** {budget/size:,.2f} SAR per m²")
st.write(f"**Worker Density:** {workers/size:.4f} workers per m²")

if budget/size < 300:
    st.info("⚠️ Low cost pressure — budget may be underestimated.")
elif budget/size > 1500:
    st.info("⚠️ High cost pressure — optimize material selection.")
