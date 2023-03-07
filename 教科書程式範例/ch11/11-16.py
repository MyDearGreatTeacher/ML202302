model_pl_svc = make_pipeline(StandardScaler(), SVC())
param = {'svc__C': 0.5, 'svc__gamma': 0.8, 'svc__kernel': 'rbf'}
model_pl_svc.set_params(**param)
print(f"觀察模型的參數設定：{model_pl_svc.get_params()['svc']}")
model_pl_svc.fit(X_train, y_train)
print('正確率為:',model_pl_svc.score(X_test, y_test).round(3))