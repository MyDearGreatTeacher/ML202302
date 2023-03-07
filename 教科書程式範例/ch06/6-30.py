from sklearn.metrics import precision_recall_curve
# precision_recall_curve的輸入參數是機率值
prec, recall, thres = precision_recall_curve(y_test, y_pred_proba, pos_label=2)
df_p_r = pd.DataFrame(zip(thres, prec, recall), columns=['門檻','精確率','召回率'])
# 判斷門檻最大的前五筆
display(df_p_r.tail())
ax = df_p_r.plot(x='召回率', y='精確率', marker='o');
for idx in df_p_r.index:
    ax.text(x=df_p_r.loc[idx,'召回率'], y=df_p_r.loc[idx,'精確率']-0.02,
            s=df_p_r.loc[idx,'門檻'].round(2))