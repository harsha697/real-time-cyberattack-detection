import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

print("Loading full datasets...")

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

print("Loading top 43 features...")
feat_df = pd.read_csv("feature_importance.csv")
top_features = feat_df["feature"].head(43).tolist()  # USE ALL 43

print("Top Features Used:", top_features)

# Select only top features by column NAME
X_train_sel = X_train[top_features]
X_test_sel = X_test[top_features]

print("Training Random Forest Model with 43 features...")
model = RandomForestClassifier(n_estimators=500, max_depth=300, random_state=42)
model.fit(X_train_sel, y_train)

# Save the trained model
joblib.dump(model, "intrusion_model.pkl")
print("Model saved as intrusion_model.pkl")

# Evaluate
y_pred = model.predict(X_test_sel)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the feature names too (important for detector)
joblib.dump(top_features, "feature_names.pkl")
print("Top features saved as feature_names.pkl")
