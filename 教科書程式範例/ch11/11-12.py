model_pl_svc = make_pipeline(StandardScaler(), SVC(C=0.5, kernel='linear'))
model_pl_svc.fit(X_train, y_train)
model_pl_svc.score(X_test, y_test)