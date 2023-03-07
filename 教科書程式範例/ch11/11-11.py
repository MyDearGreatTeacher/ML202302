from sklearn.model_selection import cross_val_score
model_pl_svc = make_pipeline(StandardScaler(), SVC(C=0.5, kernel='linear'))
scores = cross_val_score(model_pl_svc, X_train, y_train, cv=10)
print(f'十折交叉驗證的預測結果：{scores.round(3)}')
print(f'十折交叉驗證結果的平均值{scores.mean().round(3)}')