import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Loading processed datasets...")

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print("Training RandomForest for Feature Importance...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Extracting feature importance...")

importances = model.feature_importances_
feature_names = list(X_train.columns)

feature_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

feature_df.to_csv("feature_importance.csv", index=False)

print("\nTop 20 Important Features:")
print(feature_df.head(20))

print("\nSaved to feature_importance.csv")
