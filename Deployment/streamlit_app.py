import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open('xgb_model.pkl', 'rb') as model_file:
    xgb = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

st.title("📦 Sales Prediction App")
st.write("Enter the input features to predict Sales")

# User inputs
delivery_time = st.number_input("Delivery Time (days)", min_value=0, max_value=30, value=4)
quantity = st.number_input("Quantity", min_value=1, max_value=100, value=2)
category = st.number_input("Category (encoded)", min_value=0, max_value=10, value=1)
sub_category = st.number_input("Sub-Category (encoded)", min_value=0, max_value=20, value=3)
discount = st.text_input("Discount (%)", value="15.0")
profit = st.text_input("Profit ($)", value="50.0")
popularity = st.number_input("Product Popularity", min_value=0, max_value=500, value=8)
order_count = st.number_input("Customer Order Count", min_value=1, max_value=100, value=2)

# Prediction
if st.button("Predict Sales"):
    try:
        # Convert inputs
        features = np.array([[delivery_time, quantity, category, sub_category,
                              float(discount), float(profit), popularity, order_count]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict log-sales
        log_sales = xgb.predict(features_scaled)

        # Reverse log1p
        actual_sales = np.expm1(log_sales)[0]

        st.success(f"💰 Predicted Sales: ${actual_sales:.2f}")

    except ValueError:
        st.error("Please enter valid numerical values for discount and profit.")
