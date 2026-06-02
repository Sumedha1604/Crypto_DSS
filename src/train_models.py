import joblib
import pandas as pd
import time

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# --------------------------------
# Load Dataset
# --------------------------------

df = pd.read_csv("data/final_btc_dataset.csv")

# --------------------------------
# Features
# --------------------------------

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

X = df[features]
y = df["Target"]

# --------------------------------
# Chronological Split
# --------------------------------

split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# --------------------------------
# Random Forest
# --------------------------------

start = time.time()

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_time = time.time() - start

rf_pred = rf.predict(X_test)

# --------------------------------
# Gradient Boosting
# --------------------------------

start = time.time()

gb = GradientBoostingClassifier(
    random_state=42
)

gb.fit(X_train, y_train)

gb_time = time.time() - start

gb_pred = gb.predict(X_test)

# --------------------------------
# Results
# --------------------------------

print("\n===== RANDOM FOREST =====")

print("Accuracy:",
      accuracy_score(y_test, rf_pred))

print("Precision:",
      precision_score(y_test, rf_pred))

print("Recall:",
      recall_score(y_test, rf_pred))

print("F1 Score:",
      f1_score(y_test, rf_pred))

print("Training Time:",
      round(rf_time, 2), "seconds")

print("\nConfusion Matrix")

print(confusion_matrix(
    y_test,
    rf_pred
))

print("\n===== GRADIENT BOOSTING =====")

print("Accuracy:",
      accuracy_score(y_test, gb_pred))

print("Precision:",
      precision_score(y_test, gb_pred))

print("Recall:",
      recall_score(y_test, gb_pred))

print("F1 Score:",
      f1_score(y_test, gb_pred))

print("Training Time:",
      round(gb_time, 2), "seconds")

print("\nConfusion Matrix")

print(confusion_matrix(
    y_test,
    gb_pred
))
# Save trained models

joblib.dump(
    rf,
    "models/random_forest.pkl"
)

joblib.dump(
    gb,
    "models/gradient_boosting.pkl"
)

print("\nModels saved!")
results = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Gradient Boosting"
    ],
    "Accuracy": [
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, gb_pred)
    ],
    "Precision": [
        precision_score(y_test, rf_pred),
        precision_score(y_test, gb_pred)
    ],
    "Recall": [
        recall_score(y_test, rf_pred),
        recall_score(y_test, gb_pred)
    ],
    "F1": [
        f1_score(y_test, rf_pred),
        f1_score(y_test, gb_pred)
    ],
    "Training_Time": [
        rf_time,
        gb_time
    ]
})

results.to_csv(
    "outputs/model_results.csv",
    index=False
)

print("\nResults saved!")