colors = ['red']*5 + ['blue']*(len(df_test)-5)

fig, axes = plt.subplots(1, 2, figsize=(12,4))
# 第一張圖
ax = axes[0]
df_test.plot(kind='scatter', x='RM', y='error', c=colors, ax=ax)
for i in df_test.index[:5]:
    ax.text(x=df_test.loc[i,'RM']+0.1, y=df_test.loc[i,'error']-1, s=i)
    ax.vlines(x=df_test.loc[i,'RM'], ymin=0, ymax=df_test.loc[i,'error'], ls=':')
ax.axhline(0, c='r', ls='--')
ax.set_title('殘差值分佈')

# 第二張圖
ax = axes[1]
df_test.plot(kind='scatter', x='RM', y='target', c=colors, ax=ax)
df_test.plot(kind='scatter', x='RM', y='y_pred', c='gray', ax=ax)
for i in df_test.index[:5]:
    ax.text(x=df_test.loc[i,'RM']+0.1, y=df_test.loc[i,'target']-1, s=i)
    ax.vlines(x=df_test.loc[i,'RM'], 
              ymin=df_test.loc[i,'target'], ymax=df_test.loc[i,'y_pred'], ls=':')
ax.set_title('實際值分佈');