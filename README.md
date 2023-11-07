# Zhalgas-Kuzhykov-CatBoost
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

try:
    data = pd.read_csv("C:/Users/XE/Downloads/populationkz.csv")
except FileNotFoundError:
    print("Файл не найден.")
    exit(1)

X = data[['latitude', 'longitude']]
y = data['population_2020'] - data['population_2015']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.1, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)
train_pool = Pool(data=X_train, label=y_train)
val_pool = Pool(data=X_val, label=y_val)

# Parameter grid with correct hyperparameter names
param_grid = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.01, 0.1, 0.3],
    'depth': [4, 6, 8]
}

grid_search = GridSearchCV(CatBoostRegressor(), param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, loss_function='RMSE', random_seed=42)
best_model.fit(X_train, y_train)

test_pool = Pool(data=X_test)
predictions = best_model.predict(test_pool)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual Values vs Predicted Values')
plt.show()

feature_importance = best_model.get_feature_importance(type='FeatureImportance')
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.show()
