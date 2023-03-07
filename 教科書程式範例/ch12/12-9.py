from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth': [1,2,3,4],
    'n_estimators': [100, 300, 500]
}
rfc = RandomForestClassifier(random_state=42)
gs = GridSearchCV(rfc, param_grid=param_grid, cv=10)
gs.fit(X_train, y_train)
print('最佳參數', gs.best_params_)
print('訓練集的預測結果', gs.best_score_)
print('測試集的預測結果', gs.best_estimator_.score(X_test, y_test))