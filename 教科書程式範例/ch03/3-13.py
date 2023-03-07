from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder(sparse=False)
X_cat_impute_oh = oh.fit_transform(X_cat_impute)
X_cat_impute_oh