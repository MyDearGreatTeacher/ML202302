from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=500, max_depth=1, learning_rate=0.05)
xgb.fit(X_train, y_train)
print('訓練集的預測結果', xgb.score(X_train, y_train))
print('測試集的預測結果', xgb.score(X_test, y_test))