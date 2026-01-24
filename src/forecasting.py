from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# Load forecast results
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent
FORECAST_PATH = BASE_DIR / "data" / "raw" / "silver_prices_forecast_2026.csv"

future_df = pd.read_csv(FORECAST_PATH)

# Convert Date
future_df['Date'] = pd.to_datetime(future_df['Date'])

print(future_df.info())

# ==============================
# Plot Forecast with Confidence Bands
# ==============================

plt.figure(figsize=(12, 6))

plt.plot(
    future_df['Date'],
    future_df['Predicted_Price'],
    label='Forecasted Price',
    color='green'
)

plt.fill_between(
    future_df['Date'],
    future_df['Lower_Bound'],
    future_df['Upper_Bound'],
    color='green',
    alpha=0.2,
    label='Confidence Interval'
)

plt.title("Silver Price Forecast (2026)")
plt.xlabel("Date")
plt.ylabel("Silver Price")
plt.legend()
plt.tight_layout()
plt.show()
