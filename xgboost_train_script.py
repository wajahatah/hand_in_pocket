import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import pandas as pd

df = pd.read_csv('C:/wajahat/hand_in_pocket/dataset/split_keypoint/combined/temp_keypoint_l1_norm.csv', dtype=str)

df = df.drop(columns=['source_file'])

df = df.replace(r'^\s*$', pd.NA, regex=True)  # Replace empty strings with NaN
df = df.dropna()  # Drop rows with NaN values

df = df.apply(pd.to_numeric)

X = df.drop(columns=['hand_in_pocket'])  # Features
y = df['hand_in_pocket']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

xgb_clf = xgb.XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss',
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=1.0,    
    # colsample_bytree=0.8,
    # gamma=0.1,
    # reg_alpha=0.1,
    # reg_lambda=1.0,
    random_state=42,
    objective='binary:logistic',
    tree_method='gpu_hist',  # Use GPU acceleration
    gpu_id=0,  # Specify the GPU ID if you have multiple GPUs
    predictor='gpu_predictor'  # Use GPU predictor
    )

# grif search logic
# xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [3, 5, 7, 10],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'gamma': [0, 0.1, 0.2],
#     'reg_alpha': [0, 0.1, 1],
#     'reg_lambda': [1, 1.5, 2]
# }

# grid = GridSearchCV(
#     estimator=xgb_clf,
#     param_grid=param_grid,
#     scoring='accuracy',
#     cv=5,
#     n_jobs=-1,
#     verbose=2
# )

xgb_model= xgb_clf.fit(X_train, y_train)

# 5. Evaluate
# print("Best Score:", grid.best_score_)
# print("Best Parameters:", grid.best_params_)

# 6. Test on test data (optional)
y_pred = xgb_model.predict(X_test)
print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. Save model
model_name = "xgboost_grid_temp_norm_l1.joblib"
# joblib.dump(grid.best_estimator_, f"rf_models/{model_name}" )
joblib.dump(xgb_model, f"rf_models/{model_name}" )

importances = xgb_model.feature_importances_
feature_names = X.columns

# Sort by importance
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Save to text file
txt_filename = model_name.replace(".joblib", "_feature_importances.txt")
feature_importance_df.to_csv(txt_filename, index=False, sep='\t')
print(f"Feature importances saved to {txt_filename}")
