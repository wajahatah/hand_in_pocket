import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.model_selection import GridSearchCV

# Load the cleaned dataset
# df = pd.read_csv('C:/wajahat/hand_in_pocket/dataset/training/combined/temp_l3.csv', dtype=str)
# df = pd.read_csv('C:/wajahat/hand_in_pocket/dataset/training/c1_v1.csv', dtype=str)
df = pd.read_csv('C:/wajahat/hand_in_pocket/dataset/split_keypoint/combined/temp_kp_l1_v2_pos_gen.csv', dtype=str)

df = df.drop(columns=['source_file']) 

df = df.replace(r'^\s*$', pd.NA, regex=True)  # Replace empty strings with NaN
df = df.dropna()  # Drop rows with NaN values

df = df.apply(pd.to_numeric)

# Separate features and target
X = df.drop(columns=['hand_in_pocket'])  # Features
y = df['hand_in_pocket']
# .reset_index(drop=True)                 # Target

# Temporal Feature logic 
# window_size = 5  # Size of the rolling window
# X_temporal =[]
# y_temporal =[]

# for i in range(len(X)-window_size+1):
#     X_window = X.iloc[i:i+window_size].values.flatten()
#     X_temporal.append(X_window)

#     y_temporal.append(y[i+window_size-1])  # Use the label of the last row in the window

# X_temporal = pd.DataFrame(X_temporal)
# y_temporal = pd.Series(y_temporal)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    # X_temporal, y_temporal, test_size=0.2, random_state=42, stratify=y_temporal
)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
# model = RandomForestClassifier(
#     n_estimators=100,       # Number of trees
#     max_depth=10,           # Limit the depth of the tree
#     min_samples_split=5,    # Minimum samples required to split an internal node
#     min_samples_leaf=2,     # Minimum samples required at a leaf node
#     random_state=42
# )
rf_model = model.fit(X_train, y_train)

# grid search logic 
# rf_grid = RandomForestClassifier()
# gr_search= {
#     'max_depth': ['None',5, 7, 10, 13],
#     'n_estimators': [75, 100, 150, 200],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2'],
#     'criterion': ['gini', 'entropy','log_loss'],
#     'random_state': [42]
# }

# grid = GridSearchCV(rf_grid, gr_search, cv=5, n_jobs=-1, scoring = 'accuracy', verbose=2)

# rf_model = grid.fit(X_train, y_train)
# print("Best parameters found: ", rf_model.best_params_)
# print("Best score found: ", rf_model.best_score_)

# y_pred = rf_model.predict(X_test)
# end of grid search logic

# Predict on test set

rf_model = model.fit(X_train, y_train)

model_name = "rf_temp_pos_gen.joblib"

# joblib.dump(rf_model.best_estimator_, f"rf_models/{model_name}")
joblib.dump(rf_model, f"rf_models/{model_name}")

y_pred = model.predict(X_test)

# Evaluate
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# importances = rf_model.best_estimator_.feature_importances_
importances = rf_model.feature_importances_
# feature_names = X.columns

# Create a DataFrame for easy viewing/sorting
feature_importance_df = pd.DataFrame({
    # 'feature': feature_names,
    'feature': X.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print(f"\n📊 Features of {model_name} by Importance:\n", feature_importance_df.head(15))

txt_filename = model_name.replace(".joblib", "_feature_importances.txt")
feature_importance_df.to_csv(f"txt/{txt_filename}", index=False, sep='\t')
print(f"Feature importances saved to {txt_filename}")