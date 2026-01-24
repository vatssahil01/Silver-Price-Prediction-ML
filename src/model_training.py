# Training logic in this file

# Train / Test Split (NO DATA LEAKING)

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load cleaned dataset
df = pd.read_csv("../data/processed/cleaned_silver_prices.csv")

# Define features and target

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

# Time-Based Train/Test Split-->
#  80 % train , 20% test


split_index = int(len(df) * 0.8)

x_train = x.iloc[:split_index]
x_test = x.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# verify split

print('Train size', x_train.shape)
print("Test size", x_test.shape)

# Initialize model

model = LinearRegression()

# Train model

model.fit(x_train, y_train)

print("Model training complete successfully.")


# Understanding Coefficients

coeff_df = pd.DataFrame({
    "Feature": x.columns,
    "Coefficient": model.coef_
})

print(coeff_df)


#  Model Intercept

print("Intercept:", model.intercept_)

# Save the trained model

joblib.dump(model, "../models/linear_regression_model.pkl")
print("Model saved successfully.")