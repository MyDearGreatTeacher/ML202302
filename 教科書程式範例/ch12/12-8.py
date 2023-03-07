from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=42, n_estimators=50)
rfc.fit(X_train, y_train)
rfc.score(X_test, y_test)
print('訓練集的預測結果', rfc.score(X_train, y_train))
print('測試集的預測結果', rfc.score(X_test, y_test))