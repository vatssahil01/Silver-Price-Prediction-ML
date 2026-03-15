### Forecasting Limitation

The Linear Regression model relies on technical indicators such as moving averages
and volatility, which are computed from historical prices. Since these features
are not directly available for future dates, the model is suitable for trend
analysis and short-term prediction, but not for true long-horizon forecasting.

For real future forecasting, dedicated time-series models such as ARIMA,
Prophet, or LSTM would be more appropriate.

