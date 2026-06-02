import joblib
import matplotlib.pyplot as plt

rf = joblib.load("models/random_forest.pkl")

features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume BTC",
    "Volume USDT",
    "tradecount",
    "MA10",
    "EMA10",
    "RSI",
    "MACD",
    "Volatility",
    "Lag1",
    "Lag2",
    "Lag3"
]

importance = rf.feature_importances_

plt.figure(figsize=(10,6), facecolor="#0E1117")

ax = plt.gca()
ax.set_facecolor("#0E1117")

plt.barh(features, importance)

plt.title(
    "Random Forest Feature Importance",
    color="white"
)

plt.xlabel(
    "Importance",
    color="white"
)

plt.xticks(color="white")
plt.yticks(color="white")

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png",
    facecolor="#0E1117"
)

print("Feature importance graph saved!")