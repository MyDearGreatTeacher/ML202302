si = SimpleImputer(strategy='constant', fill_value='Missing')
X_cat_impute = si.fit_transform(X_cat)
X_cat_impute