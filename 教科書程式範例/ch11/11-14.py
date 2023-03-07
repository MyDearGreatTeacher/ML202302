model_pl_svc = make_pipeline(StandardScaler(), SVC())
model_pl_svc.fit(X_train, y_train)
print(f'支持向量機的正確率：{model_pl_svc.score(X_test, y_test).round(3)}')