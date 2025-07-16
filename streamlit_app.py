import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Load trained model and scaler
model = joblib.load("svc_model.joblib")
scaler = joblib.load("scaler.joblib")

st.set_page_config(page_title="Quantum Bullish Predictor", layout="centered")
st.title("🔮 Quantum Bullish Stock Predictor")

stock = st.text_input("Enter NSE Stock Symbol (e.g., TCS.NS)", value="TCS.NS")

if st.button("Analyze"):
    if not stock or "." not in stock:
        st.error("Please enter a valid NSE stock symbol like 'TCS.NS'")
        st.stop()

    try:
        df = yf.download(stock, period='6mo', interval='1d')
        if df.empty:
            st.error("No data found for this symbol. Please try another.")
            st.stop()
    except Exception as e:
        st.error("Error fetching stock data.")
        st.code(str(e))
        st.stop()

if st.button("Analyze"):
    try:
        df = yf.download(stock, period='6mo', interval='1d')
except Exception as e:
    st.error("Failed to fetch stock data. Check your internet or stock symbol.")
    st.stop()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean()))
        df['Volatility'] = df['Close'].rolling(window=10).std()
        df.dropna(inplace=True)

        latest = df.iloc[-1]
        features = [latest['MA_10'], latest['RSI'], latest['Volatility'], latest['Close']]
        X_scaled = scaler.transform([features])

        prob = model.predict_proba(X_scaled)[0][1]
        prediction = model.predict(X_scaled)[0]

        signal = "Buy" if prediction == 1 else "Hold"
        confidence = prob if prediction == 1 else 1 - prob
        target = features[-1] * 1.04
        stoploss = features[-1] * 0.97

        st.subheader(f"📊 Prediction for {stock}")
        st.markdown(f"**Signal:** {signal}")
        st.markdown(f"**Confidence:** {round(confidence*100, 2)}%")
        st.markdown(f"**Target Price:** ₹{round(target, 2)}")
        st.markdown(f"**Stoploss:** ₹{round(stoploss, 2)}")

        st.line_chart(df[['Close', 'MA_10']].tail(60))
    except Exception as e:
        st.error("Failed to fetch or process stock data.")
        st.code(str(e))
