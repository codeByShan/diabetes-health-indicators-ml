import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Health Indicators",
    page_icon="🩺",
    layout="wide"
)

# ── Load models & scaler ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    binary_model      = joblib.load("models/binary_model.joblib")
    multiclass_model  = joblib.load("models/multiclass_model.joblib")
    regression_model  = joblib.load("models/regression_model.joblib")
    scaler            = joblib.load("models/scaler.joblib")
    return binary_model, multiclass_model, regression_model, scaler

binary_model, multiclass_model, regression_model, scaler = load_models()

# ── Label map for multiclass ──────────────────────────────────────────────────
stage_map = {0: "Gestational", 1: "No Diabetes", 2: "Pre-Diabetes", 3: "Type 1", 4: "Type 2"}

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🩺 Diabetes Health Indicators — ML Prediction App")
st.markdown("Use the sidebar to enter patient details and get predictions across three tasks.")

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("📋 Patient Information")

age                              = st.sidebar.slider("Age", 18, 90, 45)
gender                           = st.sidebar.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
ethnicity                        = st.sidebar.selectbox("Ethnicity", [0, 1, 2, 3, 4])
education_level                  = st.sidebar.selectbox("Education Level", [0, 1, 2, 3])
income_level                     = st.sidebar.selectbox("Income Level", [0, 1, 2, 3])
employment_status                = st.sidebar.selectbox("Employment Status", [0, 1, 2])
smoking_status                   = st.sidebar.selectbox("Smoking Status", [0, 1, 2])
alcohol_consumption_per_week     = st.sidebar.slider("Alcohol Consumption (per week)", 0, 20, 2)
physical_activity_minutes_per_week = st.sidebar.slider("Physical Activity (mins/week)", 0, 600, 150)
diet_score                       = st.sidebar.slider("Diet Score", 0.0, 10.0, 5.0)
sleep_hours_per_day              = st.sidebar.slider("Sleep Hours/Day", 3.0, 12.0, 7.0)
screen_time_hours_per_day        = st.sidebar.slider("Screen Time Hours/Day", 0.0, 16.0, 4.0)
family_history_diabetes          = st.sidebar.selectbox("Family History of Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
hypertension_history             = st.sidebar.selectbox("Hypertension History", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
cardiovascular_history           = st.sidebar.selectbox("Cardiovascular History", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
bmi                              = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
waist_to_hip_ratio               = st.sidebar.slider("Waist to Hip Ratio", 0.5, 2.0, 0.9)
systolic_bp                      = st.sidebar.slider("Systolic BP", 80, 200, 120)
diastolic_bp                     = st.sidebar.slider("Diastolic BP", 50, 130, 80)
heart_rate                       = st.sidebar.slider("Heart Rate", 40, 130, 75)
cholesterol_total                = st.sidebar.slider("Total Cholesterol", 100, 400, 200)
hdl_cholesterol                  = st.sidebar.slider("HDL Cholesterol", 20, 100, 50)
ldl_cholesterol                  = st.sidebar.slider("LDL Cholesterol", 50, 300, 120)
triglycerides                    = st.sidebar.slider("Triglycerides", 50, 500, 150)
glucose_fasting                  = st.sidebar.slider("Fasting Glucose", 50, 300, 100)
glucose_postprandial             = st.sidebar.slider("Postprandial Glucose", 50, 400, 140)
insulin_level                    = st.sidebar.slider("Insulin Level", 1.0, 100.0, 10.0)
hba1c                            = st.sidebar.slider("HbA1c (%)", 3.0, 15.0, 5.5)

# ── Assemble input ────────────────────────────────────────────────────────────
input_data = np.array([[age, gender, ethnicity, education_level, income_level,
                        employment_status, smoking_status, alcohol_consumption_per_week,
                        physical_activity_minutes_per_week, diet_score, sleep_hours_per_day,
                        screen_time_hours_per_day, family_history_diabetes, hypertension_history,
                        cardiovascular_history, bmi, waist_to_hip_ratio, systolic_bp,
                        diastolic_bp, heart_rate, cholesterol_total, hdl_cholesterol,
                        ldl_cholesterol, triglycerides, glucose_fasting, glucose_postprandial,
                        insulin_level, hba1c]])

input_scaled = scaler.transform(input_data)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔴 Binary Classification",
    "🟡 Multiclass Classification",
    "🔵 Regression"
])

# ── Tab 1: Binary ─────────────────────────────────────────────────────────────
with tab1:
    st.header("🔴 Binary Classification: Diabetes Diagnosis")
    st.markdown("Predicts whether the patient **has diabetes or not**.")

    if st.button("🔍 Predict Diabetes (Yes/No)", key="binary"):
        prediction = binary_model.predict(input_scaled)[0]
        probability = binary_model.predict_proba(input_scaled)[0]

        if prediction == 1:
            st.error(f"⚠️ Result: **Diabetic (Yes)**")
        else:
            st.success(f"✅ Result: **Not Diabetic (No)**")

        st.metric("Confidence - Not Diabetic", f"{probability[0]*100:.2f}%")
        st.metric("Confidence - Diabetic",     f"{probability[1]*100:.2f}%")

# ── Tab 2: Multiclass ─────────────────────────────────────────────────────────
with tab2:
    st.header("🟡 Multiclass Classification: Diabetes Stage")
    st.markdown("Predicts the **stage of diabetes** the patient is in.")

    if st.button("🔍 Predict Diabetes Stage", key="multiclass"):
        prediction = multiclass_model.predict(input_scaled)[0]
        stage = stage_map.get(prediction, "Unknown")

        st.info(f"📊 Predicted Stage: **{stage}**")

        probabilities = multiclass_model.predict_proba(input_scaled)[0]
        prob_df = pd.DataFrame({
            'Stage': list(stage_map.values()),
            'Probability': probabilities
        }).sort_values('Probability', ascending=False)

        st.bar_chart(prob_df.set_index('Stage'))

# ── Tab 3: Regression ─────────────────────────────────────────────────────────
with tab3:
    st.header("🔵 Regression: Diabetes Risk Score")
    st.markdown("Predicts the **continuous diabetes risk score** for the patient.")

    if st.button("🔍 Predict Risk Score", key="regression"):
        prediction = regression_model.predict(input_scaled)[0]

        st.metric("Predicted Diabetes Risk Score", f"{prediction:.2f}")

        if prediction < 20:
            st.success("🟢 Low Risk")
        elif prediction < 40:
            st.warning("🟡 Moderate Risk")
        else:
            st.error("🔴 High Risk")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("Built with ❤️ using Scikit-learn & Streamlit | Diabetes Health Indicators ML Project")