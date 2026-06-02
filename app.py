import streamlit as st
import pandas as pd

# ----------------------------------
# Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="Crypto DSS",
    page_icon="📈",
    layout="wide"
)

# ----------------------------------
# Load Results
# ----------------------------------

results_df = pd.read_csv(
    "outputs/model_results.csv"
)

rf = results_df.iloc[0]
gb = results_df.iloc[1]

# ----------------------------------
# KPI Calculations
# ----------------------------------

best_accuracy = max(
    rf["Accuracy"],
    gb["Accuracy"]
)

best_f1 = max(
    rf["F1"],
    gb["F1"]
)

recommended = (
    "Gradient Boosting"
    if gb["F1"] > rf["F1"]
    else "Random Forest"
)

# ----------------------------------
# Title
# ----------------------------------

st.title("📈 Cryptocurrency Decision Support System")

st.subheader(
    "Bitcoin Price Movement Prediction using Machine Learning"
)

# ----------------------------------
# KPI Cards
# ----------------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Best Accuracy",
    f"{best_accuracy:.2%}"
)

col2.metric(
    "Best F1 Score",
    f"{best_f1:.2%}"
)

col3.metric(
    "Recommended Model",
    recommended
)

# ----------------------------------
# Model Comparison
# ----------------------------------

st.header("📊 Model Comparison")
display_df = results_df.copy()

display_df["Accuracy"] = (
    display_df["Accuracy"] * 100
).round(2).astype(str) + "%"

display_df["Precision"] = (
    display_df["Precision"] * 100
).round(2).astype(str) + "%"

display_df["Recall"] = (
    display_df["Recall"] * 100
).round(2).astype(str) + "%"

display_df["F1"] = (
    display_df["F1"] * 100
).round(2).astype(str) + "%"

st.dataframe(
    display_df,
    use_container_width=True
)
# ----------------------------------
# DSS Recommendation
# ----------------------------------

st.header("🧠 Decision Support Recommendation")

if gb["F1"] > rf["F1"]:

    st.success(
        "Recommended Model: Gradient Boosting"
    )

    st.markdown("""
### Why?

- Higher Accuracy
- Higher Precision
- Higher Recall
- Higher F1 Score
- Better overall classification performance
""")

else:

    st.success(
        "Recommended Model: Random Forest"
    )

    st.markdown("""
### Why?

- Better overall performance
- Faster training time
""")

# ----------------------------------
# Dataset Information
# ----------------------------------

st.header("📂 Dataset Information")

st.markdown("""
### Dataset

- Binance BTCUSDT Hourly Dataset
- 53,962 records
- Historical OHLCV market data

### Features Used

- Open
- High
- Low
- Close
- Volume BTC
- Volume USDT
- Trade Count
- MA10
- EMA10
- RSI
- MACD
- Volatility
- Lag1
- Lag2
- Lag3
""")
# ----------------------------------
# Feature Importance
# ----------------------------------

st.header("📈 Feature Importance")

st.image(
    "outputs/feature_importance.png",
    use_container_width=True
)
# ----------------------------------
# DSS Workflow
# ----------------------------------

st.header("⚙️ DSS Workflow")

st.markdown("""
1. Collect Bitcoin historical data  
2. Preprocess and clean data  
3. Generate technical indicators  
4. Train Random Forest model  
5. Train Gradient Boosting model  
6. Evaluate model performance  
7. Compare evaluation metrics  
8. Recommend the best model using DSS logic  
""")
# ----------------------------------
# Project Summary
# ----------------------------------

st.header("📑 Project Summary")

st.markdown("""
### Objective

This Decision Support System (DSS) assists users in selecting the most suitable machine learning model for short-term Bitcoin price movement prediction.

### Machine Learning Models

1. Random Forest Classifier
2. Gradient Boosting Classifier

### Evaluation Criteria

- Accuracy
- Precision
- Recall
- F1 Score
- Training Time

### Decision Support Process

1. Load Bitcoin market data
2. Perform feature engineering
3. Train machine learning models
4. Evaluate model performance
5. Compare metrics
6. Recommend the best model

### Final Recommendation

Based on the current evaluation metrics, the system recommends the model with the highest overall predictive performance.
""")

# ----------------------------------
# Footer
# ----------------------------------

st.markdown("---")

st.caption(
    "Decision Support System for Cryptocurrency Price Prediction | BTH Software Engineering Project"
)