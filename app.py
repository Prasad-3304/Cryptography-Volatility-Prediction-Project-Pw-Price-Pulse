import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# 1. Load the trained model
if os.path.exists('volatility_model.pkl'):
    model = joblib.load('volatility_model.pkl')
else:
    st.error("Model file 'volatility_model.pkl' not found! Please place it in this folder.")
    st.stop()

st.title("PricePulse: Crypto Volatility Predictor")
st.write("Upload your OHLC data to forecast market stability.")

# 2. File Uploader
uploaded_file = st.file_uploader("Choose your dataset.csv file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    # 3. Feature Engineering (Creating missing columns)
    # Ensure column names match your CSV (e.g., 'close' or 'Close')
    data['MA_20'] = data['close'].rolling(window=20).mean()
    data['liquidity_ratio'] = data['volume'] / ((data['high'] - data['low']) + 1e-10)
    
    # 4. Cleaning (Drop NaNs created by MA_20)
    data = data.dropna(subset=['open', 'high', 'low', 'volume', 'MA_20', 'liquidity_ratio'])
    
    # 5. Make Prediction
    features = data[['open', 'high', 'low', 'volume', 'MA_20', 'liquidity_ratio']]
    prediction = model.predict(features)
    
   # 6. Display Results
    st.subheader("Market Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Volatility", f"{prediction.mean():.2e}")
    col2.metric("Max Volatility", f"{prediction.max():.2e}")
    col3.metric("Data Points", len(prediction))

    st.subheader("Predicted Volatility Trends")
    st.line_chart(prediction)
    
    st.write("Recent Predictions (Last 10 Rows):", pd.DataFrame(prediction, columns=['Predicted Volatility']).tail(10))
    st.success("Analysis Complete!")