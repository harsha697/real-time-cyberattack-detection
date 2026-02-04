import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("Loading datasets...")

train_path = "/home/robotics/MLCyberProject/capstone_project/UNSW_NB15_training-set(in).csv"
test_path  = "/home/robotics/MLCyberProject/capstone_project/UNSW_NB15_testing-set(in).csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Datasets loaded!")
print("Train:", train.shape, " Test:", test.shape)

# Combine datasets for consistent Label Encoding
combined = pd.concat([train, test], axis=0, ignore_index=True)

categorical_cols = ['proto', 'service', 'state', 'attack_cat']
encoders = {}

print("Encoding categorical columns...")

for col in categorical_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))
    encoders[col] = le

print("Encoding Complete!")

# Split back into train and test
train_encoded = combined.iloc[:len(train)]
test_encoded = combined.iloc[len(train):]

# ---- IMPORTANT: remove attack_cat (text version of label) ----
X_train = train_encoded.drop(['label', 'attack_cat'], axis=1)
y_train = train_encoded['label']

X_test = test_encoded.drop(['label', 'attack_cat'], axis=1)
y_test = test_encoded['label']

print("Final feature count:", X_train.shape[1])

# Scale numeric columns
print("Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Save processed data **with original column names preserved**
X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_df  = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("Saving processed data...")

X_train_df.to_csv("X_train.csv", index=False)
X_test_df.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Preprocessing Completed Successfully!")
print("Files saved:")
print(" - X_train.csv")
print(" - X_test.csv")
print(" - y_train.csv")
print(" - y_test.csv")
