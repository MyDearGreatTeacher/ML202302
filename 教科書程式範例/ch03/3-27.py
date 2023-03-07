# 第三步：將所有欄位整理到DataFrame裡
X_col_cat_oh = data_pl.named_transformers_['cat_pl'].\
named_steps['onehotencoder'].get_feature_names(X_col_cat)
columns = X_col_num + X_col_cat_oh.tolist()
print('整合後的欄位資料：',columns)
pd.DataFrame(data_pl.fit_transform(X), columns=columns)