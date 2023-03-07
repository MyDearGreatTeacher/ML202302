from sklearn.metrics import roc_curve
fpr, tpr, thres = roc_curve(y_test, y_pred_proba, pos_label=2)
df_roc = pd.DataFrame(zip(thres, fpr, tpr), columns=['門檻','1-特異度','敏感度'])
display(df_roc.head())
ax = df_roc.plot(x='1-特異度', y='敏感度', marker='o')
for idx in df_roc.index:
    ax.text(x=df_roc.loc[idx,'1-特異度'], y=df_roc.loc[idx,'敏感度']-0.05,
            s=df_roc.loc[idx,'門檻'].round(2))