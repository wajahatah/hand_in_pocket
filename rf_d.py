import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the cleaned dataset
df = pd.read_csv('C:/wajahat/hand_in_pocket/dataset/training/combined/combined_1.csv', dtype=str)
# df = pd.read_csv('C:/wajahat/hand_in_pocket/dataset/split_distance/combined/combined_1.csv', dtype=str)

df = df.replace(r'^\s*$', pd.NA, regex=True)  # Replace empty strings with NaN
df = df.dropna()  # Drop rows with NaN values

df = df.apply(pd.to_numeric)

# Separate features and target
X = df.drop(columns=['hand_in_pocket'])  # Features
y = df['hand_in_pocket'].reset_index(drop=True)                 # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model = model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

joblib.dump(rf_model, "rf_models/rf_3.joblib")

# Evaluate
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame for easy viewing/sorting
fi_3 = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("\n📊 Top 10 Features by Importance:\n", fi_3.head(10))