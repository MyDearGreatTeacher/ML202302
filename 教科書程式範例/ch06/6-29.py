ax = df_p_r.plot(x='召回率', y='精確率', marker='o') 
ax.set_xlabel('類別2召回率')
ax.set_ylabel('類別2精確率')
for idx in df_p_r.index:
    ax.text(x=df_p_r.loc[idx,'召回率'], y=df_p_r.loc[idx,'精確率']-0.02,
            s=df_p_r.loc[idx,'門檻'].round(1))