rf_accuracy = 0.5123
rf_f1 = 0.5366
rf_time = 12.02

gb_accuracy = 0.5265
gb_f1 = 0.5522
gb_time = 13.08

if gb_accuracy > rf_accuracy and gb_f1 > rf_f1:
    recommended_model = "Gradient Boosting"

    reason = """
Higher Accuracy
Higher F1 Score
Better Classification Performance
"""

else:
    recommended_model = "Random Forest"

    reason = """
Lower Training Time
Competitive Performance
"""

print("\n===== DSS RECOMMENDATION =====")

print(f"\nRecommended Model: {recommended_model}")

print("\nReason:")

print(reason)