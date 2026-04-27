import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ---------------- LOAD ARTIFACTS ---------------- #
model = joblib.load('../models/model.pkl')
scaler = joblib.load('../models/scaler.pkl')
columns = joblib.load('../models/columns.pkl')
numeric_cols = joblib.load('../models/numeric_cols.pkl')
means = pd.read_csv('../models/means.csv', index_col=0).iloc[:, 0]

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

st.title("💳 Credit Risk Prediction Dashboard")

# ---------------- SIDEBAR INPUT ---------------- #
st.sidebar.header("📋 Customer Details")

age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.number_input("Income(In LPA)", min_value=10000, value=500000)
loan_amount = st.sidebar.number_input("Loan Amount(In Lakhs)", min_value=1000, value=200000)
credit_utilization_ratio = st.sidebar.slider("Credit Utilization (%)", 0, 100, 30)
loan_tenure = st.sidebar.slider("Loan Tenure (months)", 6, 60, 24)
bank_balance = st.sidebar.number_input("Bank Balance(In Lakhs)", min_value=0, value=50000)
enquiry_count = st.sidebar.slider("Enquiry Count", 1, 10, 3)

gender = st.sidebar.selectbox("Gender", ["M", "F"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married"])
employment_status = st.sidebar.selectbox("Employment", ["Salaried", "Self-Employed"])
residence_type = st.sidebar.selectbox("Residence", ["Owned", "Rented", "Mortgage"])
loan_type = st.sidebar.selectbox("Loan Type", ["Secured", "Unsecured"])
loan_purpose = st.sidebar.selectbox("Loan Purpose", ["Personal", "Home", "Auto", "Education"])

# ---------------- FEATURE ENGINEERING ---------------- #
loan_to_income = loan_amount / income
credit_utilization_per_income = credit_utilization_ratio / loan_to_income

# ---------------- CREATE INPUT DF ---------------- #
input_dict = {
    'age': age,
    'income': income,
    'loan_amount': loan_amount,
    'credit_utilization_ratio': credit_utilization_ratio,
    'loan_tenure_months': loan_tenure,
    'bank_balance_at_application': bank_balance,
    'enquiry_count': enquiry_count,
    'loan_to_income': loan_to_income,
    'credit_utilization_per_income': credit_utilization_per_income,
    'gender': gender,
    'marital_status': marital_status,
    'employment_status': employment_status,
    'residence_type': residence_type,
    'loan_type': loan_type,
    'loan_purpose': loan_purpose
}

input_df = pd.DataFrame([input_dict])

# ---------------- ENCODING ---------------- #
input_encoded = pd.get_dummies(input_df)

# ---------------- ALIGN FEATURES ---------------- #
input_encoded = input_encoded.reindex(columns=columns, fill_value=np.nan)

# ---------------- FILL MISSING ---------------- #
for col in columns:
    if input_encoded[col].isnull().any():
        input_encoded[col] = input_encoded[col].fillna(means[col])

# ---------------- SCALE ---------------- #
input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])

# ---------------- PREDICTION ---------------- #
if st.button("🚀 Predict Risk"):

    prob = model.predict_proba(input_encoded)[0][1]

    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    # --------- RISK TEXT --------- #
    with col1:
        st.metric("Default Probability", f"{prob:.2%}")

        if prob > 0.7:
            st.error("🔴 High Risk")
        elif prob > 0.4:
            st.warning("🟡 Medium Risk")
        else:
            st.success("🟢 Low Risk")

    # --------- RISK METER --------- #
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Risk Score (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"},
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    # --------- FEATURE SUMMARY --------- #
    st.subheader("📋 Input Summary")

    st.dataframe(input_df)

    # --------- INSIGHTS --------- #
    st.subheader("🧠 Model Insights")

    if prob > 0.7:
        st.write("High credit utilization and financial stress indicators detected.")
    elif prob > 0.4:
        st.write("Moderate risk. Monitor financial behavior.")
    else:
        st.write("Customer appears financially stable.")