from sklearn.preprocessing import MinMaxScaler
num_pl = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
num_pl.set_params(standardscaler=MinMaxScaler())
num_pl.fit_transform(X_num)