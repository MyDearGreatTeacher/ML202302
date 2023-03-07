from sklearn.model_selection import cross_val_score
model_pl_lr = make_pipeline(StandardScaler(), LogisticRegression())
scores = cross_val_score(model_pl_lr, X_train, y_train, scoring='accuracy', cv=5)
print(f'5折交叉驗證的每次結果 {scores}')
print(f'5折交叉驗證的平均結果{np.mean(scores)}')