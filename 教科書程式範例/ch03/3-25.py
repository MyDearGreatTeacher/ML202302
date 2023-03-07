# 第二步：取得onehotencoder欄位對應結果
data_pl.named_transformers_['cat_pl'].\
named_steps['onehotencoder'].get_feature_names()