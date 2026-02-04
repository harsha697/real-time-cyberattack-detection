import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import xgboost as xgb

# ---------------- LOAD DATA ----------------
print("Loading processed datasets...")
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# ---------------- TRAIN MODELS ----------------
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=700,
    max_depth=40,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=10,
    learning_rate=0.07,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1),
    objective='binary:logistic',
    eval_metric='logloss',
    n_jobs=-1,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# ---------------- ENSEMBLE ----------------
print("Creating soft-voting ensemble...")
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb_model)],
    voting='soft',
    n_jobs=-1
)
ensemble.fit(X_train, y_train)
joblib.dump(ensemble, "intrusion_model_ensemble.pkl")
print("Ensemble model saved as 'intrusion_model_ensemble.pkl'")

# ---------------- THRESHOLD OPTIMIZATION ----------------
print("Optimizing threshold based on F1-score...")
probs = ensemble.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.01)
best_thresh = 0.5
best_f1 = 0

for t in thresholds:
    preds = (probs >= t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"Best threshold based on F1-score: {best_thresh:.2f} (F1={best_f1:.3f})")
joblib.dump(best_thresh, "ensemble_threshold.pkl")
print("Threshold saved as 'ensemble_threshold.pkl'")

# ---------------- EVALUATE ----------------
y_pred = (probs >= best_thresh).astype(int)
print("\n===== ENSEMBLE MODEL EVALUATION =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
