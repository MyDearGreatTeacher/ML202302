rom sklearn.preprocessing import OneHotEncoder
cat_pl = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(sparse=False)
)
# 檢查類別管道器的運作  
cat_pl.fit_transform(df[X_col_cat])[:3]