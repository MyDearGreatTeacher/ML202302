model_pl = Pipeline([
    ('preprocess', data_pl),
    ('model', LogisticRegression())
])
np.random.seed(42)
param_grid = {'model':[RandomForestClassifier(), AdaBoostClassifier(), 
                       BaggingClassifier(), XGBClassifier()]}
gs = GridSearchCV(model_pl, param_grid=param_grid,
                  cv=5, return_train_score=True)
gs.fit(X_train, y_train)
score = gs.best_estimator_.score(X_test, y_test)
print('最佳預測參數', gs.best_params_)
print('訓練集交叉驗證的最佳結果', gs.best_score_.round(3))
print('測試集的結果', score.round(3))
y_pred = gs.best_estimator_.predict(X_test)
print('混亂矩陣\n',confusion_matrix(y_test, y_pred))