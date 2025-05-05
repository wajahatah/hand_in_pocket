import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib

df = pd.read_csv('C:/wajahat/hand_in_pocket/dataset/split_keypoint/combined/temp_keypoint_l1_norm.csv', dtype=str)

df = df.drop(columns=['source_file'])

df = df.replace(r'^\s*$', pd.NA, regex=True)  # Replace empty strings with NaN
df = df.dropna()  # Drop rows with NaN values

df = df.apply(pd.to_numeric)

X = df.drop(columns=['hand_in_pocket'])  # Features
y = df['hand_in_pocket']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LightGBM model
lgbm = lgb.LGBMClassifier(
    objective='binary', 
    boosting_type='gbdt', 
    random_state=42,
    n_estimators=100,
    learning_rate=0.7,
    num_leaves=31,
    max_depth=-1, 
    device='gpu')

# Define the hyperparameter grid
# param_grid = {
#     'num_leaves': [15, 31, 50],
#     'max_depth': [-1, 5, 10],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [50, 100, 200],
#     'min_child_samples': [10, 20, 30],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
# }

# # Perform grid search with cross-validation
# print("Starting Grid Search...")
# grid = GridSearchCV(estimator=lgbm,
#                     param_grid=param_grid,
#                     cv=5,
#                     scoring='accuracy',
#                     verbose=1,
#                     n_jobs=-1)

lgbm_model = lgbm.fit(X_train, y_train)

# Get the best model
# best_model = grid.best_estimator_

# Print best parameters
# print("Best Parameters Found:")
# print(grid.best_params_)

# Evaluate on the test set
y_pred = lgbm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the best model
model_filename = "lightgbm_temp_norm_l1.joblib"
joblib.dump(lgbm_model, model_filename)
print(f"Model saved to {model_filename}")

# --- Save Feature Importances ---
importances = lgbm_model.feature_importances_
feature_names = X.columns

# Sort by importance
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Save to text file
txt_filename = model_filename.replace(".joblib", "_feature_importances.txt")
feature_importance_df.to_csv(txt_filename, index=False, sep='\t')
print(f"Feature importances saved to {txt_filename}")
