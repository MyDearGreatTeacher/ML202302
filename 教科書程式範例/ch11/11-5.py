model_pl = make_pipeline(StandardScaler(), SVC())
keys = model_pl.get_params().keys()
keys