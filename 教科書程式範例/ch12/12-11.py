ada_clf = AdaBoostClassifier(RandomForestClassifier(max_depth=1, random_state=42),
                            n_estimators=500, random_state=42)
ada_clf.fit(X_train, y_train)
print('訓練集的預測結果', ada_clf.score(X_train, y_train))
print('測試集的預測結果', ada_clf.score(X_test, y_test))