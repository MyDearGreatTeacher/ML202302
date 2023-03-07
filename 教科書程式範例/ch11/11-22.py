from sklearn.preprocessing import MinMaxScaler
model_pl = Pipeline([
    ('preprocess', StandardScaler()),
    ('model', LogisticRegression())
])
preprocess = [StandardScaler(), MinMaxScaler(), None]
param_grid = [
    {'preprocess': preprocess,
     'model':[SVC()], 'model__kernel':['linear','rbf'], 
     'model__C': [0.1, 0.5, 0.8, 1, 5],'model__gamma': np.arange(0.2, 1, 0.2)},
]
gs = GridSearchCV(model_pl, param_grid=param_grid, 
                  cv=5, return_train_score=True)
gs.fit(X_train, y_train)
score = gs.best_estimator_.score(X_test, y_test)
print('最佳預處理方式', gs.best_params_['preprocess'])
print('訓練集交叉驗證的最佳結果', gs.best_score_)
print('測試集的預測結果', score)