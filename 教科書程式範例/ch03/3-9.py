from sklearn.pipeline import make_pipeline
num_pl = make_pipeline(SimpleImputer(strategy='mean'), 
                       StandardScaler())
num_pl.fit_transform(X_num)