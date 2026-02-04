import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

print("Loading model and data...")

# Load model and feature list
model = joblib.load("intrusion_model.pkl")
top_features = joblib.load("feature_names.pkl")

# Load test data
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()

# 🔴 IMPORTANT: same feature selection
X_test = X_test[top_features]

# Predict probabilities
probs = model.predict_proba(X_test)[:, 1]

# 🔧 Change threshold here
THRESHOLD = 0.25
y_pred = (probs > THRESHOLD).astype(int)

print(f"\n===== THRESHOLD EVALUATION (threshold={THRESHOLD}) =====")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
