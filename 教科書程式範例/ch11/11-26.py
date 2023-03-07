from sklearn.model_selection import RandomizedSearchCV
model_pl = Pipeline([
    ('preprocess', StandardScaler()),
    ('model', LogisticRegression())
])
param_grid = {'model':[SVC()], 'model__kernel':['linear','rbf'], 
              'model__C': [0.1, 1, 2, 3, 4, 5, 6, 7, 10, 100],
              'model__gamma': [1, 0.1, 0.01, 0.001, 0.002, 0.0001]}
n_iter = 20
random_gs = RandomizedSearchCV(model_pl, param_distributions=param_grid,
                               n_iter=n_iter, cv=5, iid=False)
random_gs.fit(X_train, y_train)
score = random_gs.best_estimator_.score(X_test, y_test)
print('最佳模型', random_gs.best_params_['model'])
print('最佳交叉驗證的結果', random_gs.best_score_)
print('最後測試集的結果', score)