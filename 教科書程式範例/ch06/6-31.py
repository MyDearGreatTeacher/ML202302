from sklearn.metrics import recall_score
scores = []
y_pred_proba = model_pl.predict_proba(X_test)[:,1]
for threshold in np.arange(0, 1, 0.1):
    y_pred = np.where(y_pred_proba>=threshold, 2, 1)
    # tpr為類別2的召回率
    tpr = recall_score(y_test, y_pred, pos_label=2)
    # fpr為類別1的召回錯誤率
    fpr = 1 - recall_score(y_test, y_pred, pos_label=1)
    scores.append([threshold, tpr, fpr])
df_roc = pd.DataFrame(scores, columns=['門檻','敏感度','1-特異度'])
df_roc.sort_values(by='門檻').head()