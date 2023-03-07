ax = df_roc.plot(x='1-特異度', y='敏感度', marker='o') 
for idx in df_roc.index:
    ax.text(x=df_roc.loc[idx,'1-特異度'], y=df_roc.loc[idx,'敏感度']-0.03,
            s=df_roc.loc[idx,'門檻'].round(1))