import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading model and data...")

# Load model and feature list
model = joblib.load("intrusion_model.pkl")
top_features = joblib.load("feature_names.pkl")

# Load test data
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()

# 🔴 IMPORTANT: select same features used in training
X_test = X_test[top_features]

print("Predicting...")
y_pred = model.predict(X_test)

print("\n===== MODEL EVALUATION =====")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
