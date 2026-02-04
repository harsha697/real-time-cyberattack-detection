import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load training data
X_train = pd.read_csv("X_train.csv")

# Specify categorical columns (same as in training)
categorical_cols = ['proto', 'service', 'state']

# Initialize LabelEncoders for categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    if col in X_train.columns:
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        label_encoders[col] = le

# Save label encoders
joblib.dump(label_encoders, "label_encoders.pkl")

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

print("Saved scaler.pkl and label_encoders.pkl successfully!")
