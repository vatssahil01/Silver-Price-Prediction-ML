# Model Evaluation

# Goal:

# Evaluate model performance

# Interpret metrics correctly

# Explain results confidently (even if they are not perfect)

from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


#  Load data & model

# Load cleaned data
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_silver_prices.csv"
MODEL_PATH = BASE_DIR / "models" / "linear_regression_model.pkl"

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)


#  Define Features and Target

x = df[
    [
        'Volume',
        'MA_50',
        'MA_200',
        'Daily_Return',
        'Volatility_30d',
        'Year',
        'Month'
    ]
]

y = df['Close']

# Time based split

split_index = int(len(df)* 0.8)

x_test = x.iloc[split_index:]
y_test = y.iloc[split_index:]


# Make Predictions

y_pred = model.predict(x_test)

#  Calculate Evaluation Matrix

mae = mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test , y_pred)

# MAE (Mean Absolute Error)
# Average absolute difference between predicted and actual prices.
print("Mean Absolute Error:", mae)

# RMSE (Root Mean Squared Error)
# Penalizes large errors more heavily.
print("Root Mean Squared Error:", rmse)

# RMSE (Root Mean Squared Error)
# Penalizes large errors more heavily.
print("R2 Score:", r2)



# Plot Actual vs Predicted Prices

plt.figure(figsize=(12, 6))

plt.plot(y_test.values, label="Actual Price", color="blue")
plt.plot(y_pred, label="Predicted Price", color="red")

plt.title("Actual vs Predicted Silver Prices")
plt.xlabel("Time Index")
plt.ylabel("Silver Price")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("../reports/actual_vs_predicted.png")
