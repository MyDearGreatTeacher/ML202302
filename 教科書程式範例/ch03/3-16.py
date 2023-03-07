cat_pl = make_pipeline(SimpleImputer(strategy='most_frequent'), 
                       OneHotEncoder(sparse=False))
cat_pl.fit_transform(X_cat)