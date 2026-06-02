import pandas as pd

# Load data
df = pd.read_csv("data/Binance_BTCUSDT_1h (1).csv")

# Convert date column
df["Date"] = pd.to_datetime(
    df["Date"],
    format="mixed"
)

# Sort oldest to newest
df = df.sort_values("Date")

# Create target variable
df["Target"] = (
    df["Close"].shift(-1) > df["Close"]
).astype(int)

# Remove last row
df = df[:-1]

print("\nDataset Shape:")
print(df.shape)

print("\nTarget Distribution:")
print(df["Target"].value_counts())

# Save processed file
df.to_csv(
    "data/processed_btc.csv",
    index=False
)

print("\nProcessed data saved!")