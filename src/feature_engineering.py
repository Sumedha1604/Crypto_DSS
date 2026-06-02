import pandas as pd
import ta

# Load processed data
df = pd.read_csv("data/processed_btc.csv")

# -----------------------------
# Moving Average
# -----------------------------
df["MA10"] = df["Close"].rolling(window=10).mean()

# -----------------------------
# Exponential Moving Average
# -----------------------------
df["EMA10"] = df["Close"].ewm(span=10).mean()

# -----------------------------
# RSI
# -----------------------------
df["RSI"] = ta.momentum.RSIIndicator(
    close=df["Close"],
    window=14
).rsi()

# -----------------------------
# MACD
# -----------------------------
macd = ta.trend.MACD(df["Close"])

df["MACD"] = macd.macd()

# -----------------------------
# Volatility
# -----------------------------
df["Volatility"] = (
    df["High"] - df["Low"]
) / df["Close"]

# -----------------------------
# Lag Features
# -----------------------------
df["Lag1"] = df["Close"].shift(1)
df["Lag2"] = df["Close"].shift(2)
df["Lag3"] = df["Close"].shift(3)

# Remove null values created
# by indicators and lag features
df.dropna(inplace=True)

print("\nFinal Shape:")
print(df.shape)

print("\nNew Features Added:")
print([
    "MA10",
    "EMA10",
    "RSI",
    "MACD",
    "Volatility",
    "Lag1",
    "Lag2",
    "Lag3"
])

# Save final dataset
df.to_csv(
    "data/final_btc_dataset.csv",
    index=False
)

print("\nFeature engineering completed!")