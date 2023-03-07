from sklearn.metrics import precision_score, recall_score
scores = []
# 先用[:,1]取得類別2的預測機率
y_pred_proba = model_pl.predict_proba(X_test)[:,1]
# 將判斷門檻從0, 0.1, 0.2, ...到1
for threshold in np.arange(0, 1, 0.1):
    # 透過np.where取得不同判斷門檻的預測結果
    y_pred = np.where(y_pred_proba>=threshold, 2, 1)
    prec = precision_score(y_test, y_pred, pos_label=2)
    recall = recall_score(y_test, y_pred, pos_label=2)
    # 將所有結果存在scores串列
    scores.append([threshold, prec, recall])
df_p_r = pd.DataFrame(scores, columns=['門檻','精確率','召回率'])
df_p_r.sort_values(by='門檻')