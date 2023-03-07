print('正確率：', accuracy_score(y_test, y_pred_8).round(2))
print('混亂矩陣')
print(pd.DataFrame(confusion_matrix(y_test, y_pred_8),
                   index=['實際1', '實際2'], columns=['預測1', '預測2']))
print('綜合報告')
print(classification_report(y_test, y_pred_8))