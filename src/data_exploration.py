import pandas as pd

# Load Bitcoin dataset
df = pd.read_csv("data/Binance_BTCUSDT_1h (1).csv")

print("\n===== First 5 Rows =====")
print(df.head())

print("\n===== Shape =====")
print(df.shape)

print("\n===== Columns =====")
print(df.columns.tolist())

print("\n===== Data Types =====")
print(df.dtypes)

print("\n===== Missing Values =====")
print(df.isnull().sum())