from sklearn.metrics import confusion_matrix
score = gs.best_estimator_.score(X_test, y_test)
y_pred = gs.best_estimator_.predict(X_test)
print(f'正確率為{score.round(3)}')
print(f'混亂矩陣結果為\n{confusion_matrix(y_test, y_pred)}')