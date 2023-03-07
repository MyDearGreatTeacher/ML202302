from sklearn.impute import SimpleImputer
si = SimpleImputer(strategy='mean')
X_num_impute = si.fit_transform(X_num)
X_num_impute