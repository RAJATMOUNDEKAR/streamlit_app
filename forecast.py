import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the transactional data
transactional_data_1 = pd.read_csv('/Users/rajat/Desktop/DataEngineer/Transactional_data_retail_01.csv')
transactional_data_2 = pd.read_csv('/Users/rajat/Desktop/DataEngineer/Transactional_data_retail_02.csv')

# Concatenate the two transactional datasets
transactional_data = pd.concat([transactional_data_1, transactional_data_2])

# Data Preprocessing
transactional_data['InvoiceDate'] = pd.to_datetime(transactional_data['InvoiceDate'], errors='coerce')
transactional_data.dropna(subset=['InvoiceDate', 'Quantity', 'Price'], inplace=True)
transactional_data.set_index('InvoiceDate', inplace=True)

# Group by StockCode and sum the Quantity sold
top_stock_codes = transactional_data.groupby('StockCode')['Quantity'].sum().reset_index()
top_10_stock_codes = top_stock_codes.sort_values(by='Quantity', ascending=False).head(10)

# Create Streamlit App
st.title("Product Demand Forecasting App")

# Display top 10 products
st.header("Top 10 Products")
top_products = top_10_stock_codes['StockCode'].tolist()
st.write(top_products)

# User input for StockCode
selected_stock_code = st.selectbox("Select a Stock Code", top_products)

# Prepare time series data for the selected product code
ts_data = transactional_data[transactional_data['StockCode'] == selected_stock_code].resample('W')['Quantity'].sum()

# Split data into train (80%) and test (20%)
train_size = int(len(ts_data) * 0.8)
train_data, test_data = ts_data[:train_size], ts_data[train_size:]

# Check if enough data points for seasonal model
seasonal_periods = 52  # For weekly data assuming yearly seasonality

if len(train_data) >= seasonal_periods * 2:
    ets_model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
else:
    ets_model = ExponentialSmoothing(train_data, trend='add')  # Non-seasonal model

ets_fit = ets_model.fit()

# Forecast the next steps (including the test period + extra future values)
forecast_steps = len(test_data) + 15  # Adding extra weeks to forecast
forecast_ets = ets_fit.forecast(steps=forecast_steps)

# Create a combined DataFrame for historical and forecast data
forecast_index = pd.date_range(start=train_data.index[-1] + pd.Timedelta(weeks=1), periods=forecast_steps, freq='W')
forecast_series = pd.Series(forecast_ets, index=forecast_index)

# Combine historical and forecast data
combined_data = pd.concat([ts_data, forecast_series])

# Plot Historical and Forecast Demand
st.subheader(f"Historical and Forecast Demand for {selected_stock_code}")
plt.figure(figsize=(10, 6))
plt.plot(ts_data.index, ts_data, label='Historical Demand', color='blue')
plt.plot(forecast_series.index, forecast_series, label='Forecasted Demand', color='red', linestyle='--')
plt.axvline(x=train_data.index[-1], color='gray', linestyle='--', label='Train-Test Split')
plt.title(f"Demand for {selected_stock_code}")
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.legend()
st.pyplot(plt)

# Calculate errors
train_forecast = ets_fit.fittedvalues
test_forecast = forecast_ets[:len(test_data)]

train_errors = train_data - train_forecast
test_errors = test_data - test_forecast

# Optional: Provide RMSE and MAE for user reference
rmse_train = np.sqrt(mean_squared_error(train_data, train_forecast))
mae_train = mean_absolute_error(train_data, train_forecast)

rmse_test = np.sqrt(mean_squared_error(test_data, test_forecast))
mae_test = mean_absolute_error(test_data, test_forecast)

st.sidebar.header("Model Performance Metrics")
st.sidebar.write(f"Training RMSE: {rmse_train:.2f}")
st.sidebar.write(f"Training MAE: {mae_train:.2f}")
st.sidebar.write(f"Testing RMSE: {rmse_test:.2f}")
st.sidebar.write(f"Testing MAE: {mae_test:.2f}")
