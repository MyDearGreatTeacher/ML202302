from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier()
ada_clf.fit(X_train, y_train)
print('訓練集的預測結果', ada_clf.score(X_train, y_train))
print('測試集的預測結果', ada_clf.score(X_test, y_test))