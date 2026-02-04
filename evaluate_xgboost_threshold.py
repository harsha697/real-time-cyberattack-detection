import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

THRESHOLD = 0.25  # try 0.2, 0.25, 0.3

print("Loading XGBoost model and data...")

model = joblib.load("intrusion_model_xgb.pkl")

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()

probs = model.predict_proba(X_test)[:, 1]
y_pred = (probs > 0.4).astype(int)

print(f"\n===== XGBOOST THRESHOLD EVALUATION (threshold={THRESHOLD}) =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
