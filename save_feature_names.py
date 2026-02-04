import pandas as pd
import joblib

# Load feature importance CSV
feat_df = pd.read_csv("feature_importance.csv")

# Get all 43 features used in training
all_features = feat_df["feature"].head(43).tolist()  # or all rows if exactly 43

# Save as pickle
joblib.dump(all_features, "feature_names.pkl")

print("Saved feature_names.pkl successfully!")
