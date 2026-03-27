import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model and encoders
model = load_model("ann_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))
le_gender = pickle.load(open("le_gender.pkl", "rb"))
ohe_geo = pickle.load(open("ohe_geo.pkl", "rb"))

# Title
st.title("Customer Churn Prediction")

# Inputs
credit_score = st.number_input("Credit Score", min_value=0)
age = st.number_input("Age", min_value=0)
tenure = st.number_input("Tenure", min_value=0)
balance = st.number_input("Balance", min_value=0.0)
num_products = st.number_input("Number of Products", min_value=1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_number = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)

gender = st.selectbox("Gender", ["Male", "Female"])
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

# Encoding
gender_encoded = le_gender.transform([gender])[0]
geo_encoded = ohe_geo.transform([[geography]])

# Combine input
input_data = np.array([[ 
    credit_score, gender_encoded, age, tenure, balance,
    num_products, has_cr_card, is_active_number, estimated_salary
]])

input_data = np.concatenate([input_data, geo_encoded], axis=1)

# Scaling
input_data = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    pred = model.predict(input_data)[0][0]

    if pred > 0.5:
        st.error(f"Churn Likely ({pred:.2f})")
    else:
        st.success(f"No Churn ({pred:.2f})")