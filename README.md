# Cryptocurrency Decision Support System (Crypto DSS)

## Overview

This project implements a Decision Support System (DSS) for Bitcoin price movement prediction using Machine Learning techniques. The system compares multiple machine learning models and recommends the most suitable model based on performance metrics.

## Objectives

* Predict short-term Bitcoin price movement.
* Compare Random Forest and Gradient Boosting classifiers.
* Evaluate model performance using multiple metrics.
* Recommend the best-performing model through a DSS framework.

## Dataset

* Binance BTCUSDT Hourly Dataset
* 53,962 observations
* Historical OHLCV (Open, High, Low, Close, Volume) data

## Feature Engineering

The following features were used:

* Moving Average (MA10)
* Exponential Moving Average (EMA10)
* RSI (Relative Strength Index)
* MACD
* Volatility
* Lag Features (Lag1, Lag2, Lag3)

## Machine Learning Models

1. Random Forest Classifier
2. Gradient Boosting Classifier

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Training Time

## Results

| Model             | Accuracy | Precision | Recall | F1 Score |
| ----------------- | -------- | --------- | ------ | -------- |
| Random Forest     | 51.23%   | 51.73%    | 55.74% | 53.66%   |
| Gradient Boosting | 52.65%   | 53.01%    | 57.63% | 55.22%   |

## DSS Recommendation

The system recommends Gradient Boosting as the preferred model due to its superior performance across evaluation metrics.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Streamlit
* Matplotlib
* Joblib

## Running the Project

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author

Sumedha Thakur
Blekinge Institute of Technology (BTH)
Master's in Software Engineering
