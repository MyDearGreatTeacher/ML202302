from sklearn.ensemble import BaggingClassifier
bagc = BaggingClassifier(random_state=42, n_estimators=50)
bagc.fit(X_train, y_train)
print('訓練集的預測結果', bagc.score(X_train, y_train))
print('測試集的預測結果', bagc.score(X_test, y_test))