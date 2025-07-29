from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import datetime
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    ticker: str

class PredictResponse(BaseModel):
    ticker: str
    latest_prediction: float

# --- ML Logic (from notebook, unchanged) ---
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def preprocess_data(stock_data):
    data = stock_data['Close'].values
    data = data.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    training_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train, scaler, scaled_data, training_data_len

def build_lstm_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)
    return model

def predict_stock_price(model, scaler, scaled_data, training_data_len):
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# --- FastAPI Endpoint ---
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    ticker = request.ticker.strip().upper()
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365 * 5)
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    if stock_data.empty:
        raise HTTPException(status_code=404, detail=f"No data found for ticker: {ticker}")
    x_train, y_train, scaler, scaled_data, training_data_len = preprocess_data(stock_data)
    model = build_lstm_model(x_train, y_train)
    predictions = predict_stock_price(model, scaler, scaled_data, training_data_len)
    latest_prediction = float(predictions[-1][0])
    return PredictResponse(ticker=ticker, latest_prediction=latest_prediction) 