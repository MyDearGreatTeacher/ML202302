kfold = KFold(n_splits=5)
model_pl_lr = make_pipeline(StandardScaler(), LogisticRegression())
scores = []
for train_idx, test_idx in kfold.split(X_train, y_train):
    model_pl_lr.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
    scores.append(model_pl_lr.score(X_train.iloc[test_idx], y_train.iloc[test_idx]))
print(f'5折交叉驗證的結果{np.mean(scores)}')