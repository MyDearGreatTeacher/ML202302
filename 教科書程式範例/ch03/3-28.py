# 資料
df_full = pd.DataFrame({'price':[10,20,30,40,10,20]})
print('原始資料\n', df_full)

from sklearn.preprocessing import KBinsDiscretizer
kb = KBinsDiscretizer(n_bins=3, encode='ordinal')
kb.fit_transform(df_full)