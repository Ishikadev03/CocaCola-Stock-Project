import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("📈 Coca-Cola Stock Price Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("Coca-Cola_stock_history.csv")
    df['Date'] = pd.to_datetime(df['Date'],errors ='coerce',infer_datetime_format=True)
    df = df.dropna(subset=['Date'])
    df.sort_values('Date', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    return df

df = load_data()

df['MA_20'] = df['Close'].rolling(20).mean()
df['MA_50'] = df['Close'].rolling(50).mean()
df['Daily_Return'] = df['Close'].pct_change()
df['Volatility'] = df['Daily_Return'].rolling(20).std()
df.dropna(inplace=True)

features = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits',
            'MA_20', 'MA_50', 'Daily_Return', 'Volatility']
X = df[features]
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


def predict_live_price():
    live_data = yf.download("KO", period='30d', interval='1d')
    live_data.reset_index(inplace=True)
    
    if 'Dividends' not in live_data.columns:
        live_data['Dividends'] = 0
    if 'Stock Splits' not in live_data.columns:
        live_data['Stock Splits'] = 0

    live_data['MA_20'] = live_data['Close'].rolling(20).mean()
    live_data['MA_50'] = live_data['Close'].rolling(50).mean()
    live_data['Daily_Return'] = live_data['Close'].pct_change()
    live_data['Volatility'] = live_data['Daily_Return'].rolling(20).std()
    live_data.fillna(0, inplace=True)

    latest_features = live_data[features].iloc[-1:]
    predicted_price = model.predict(latest_features)[0]
    return predicted_price

predicted_close = predict_live_price()

st.subheader("📊 Close Price with Moving Averages")
st.line_chart(df[['Close', 'MA_20', 'MA_50']])

st.subheader("🔮 Predicted Live Closing Price")
st.metric(label="Predicted Close", value=f"${predicted_close:.2f}")
