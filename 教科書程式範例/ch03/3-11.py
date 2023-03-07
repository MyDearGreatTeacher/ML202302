si = SimpleImputer(strategy='most_frequent')
X_cat_impute = si.fit_transform(X_cat)
X_cat_impute