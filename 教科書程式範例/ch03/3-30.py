num_pl = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
num_pl.set_params(standardscaler=None)
num_pl.fit_transform(X_num)