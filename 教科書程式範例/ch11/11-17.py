def predict(param):
    model_pl_svc = make_pipeline(StandardScaler(), SVC())
    model_pl_svc.set_params(**param)
    model_pl_svc.fit(X_train, y_train)
    return model_pl_svc.score(X_test, y_test)
df_cv['accuracy'] = df_cv['params'].apply(predict)
df_cv