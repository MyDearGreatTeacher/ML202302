from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=['實際1', '實際2'], columns=['預測1', '預測2']))
print()
print('整體正確率:', accuracy_score(y_test, y_pred).round(2))
# 另一個快速得到正確率的方法
print('另一個得到正確率的方法', model.score(X_test, y_test).round(2))