from sklearn.ensemble import GradientBoostingClassifier
gbc_clf = GradientBoostingClassifier(n_estimators=500)
gbc_clf.fit(X_train, y_train)
print('訓練集的預測結果', gbc_clf.score(X_train, y_train))
print('測試集的預測結果', gbc_clf.score(X_test, y_test))