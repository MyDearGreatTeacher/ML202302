from sklearn.pipeline import Pipeline
model_pl = Pipeline([
    ('preprocess', StandardScaler()),
    ('model', LogisticRegression())
])
param_grid = [
    {'model':[RandomForestClassifier()], 
     'model__n_estimators': [100, 500]},
    {'model':[AdaBoostClassifier()], 
     'model__n_estimators': [100, 500],
     'model__base_estimator':[None, RandomForestClassifier(max_depth=1)]},
    {'model':[XGBClassifier()], 
     'model__n_estimators': [100, 500]},
]
gs = GridSearchCV(model_pl, param_grid=param_grid,
                  cv=5, return_train_score=True)
gs.fit(X_train, y_train)
score = gs.best_estimator_.score(X_test, y_test)
print('最佳模型', gs.best_params_['model'])
print('訓練集的最佳結果', gs.best_score_)
print('測試集的結果', score)