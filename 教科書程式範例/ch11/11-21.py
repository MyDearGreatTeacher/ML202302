from sklearn.pipeline import Pipeline
model_pl = Pipeline([
    ('preprocess', StandardScaler()),
    ('model', LogisticRegression())
])

param_grid = [
    {'model':[LogisticRegression()], 'model__penalty': ['l1', 'l2'], 
     'model__C':[0.001,0.01,1,5,10]},
    {'model':[SVC()], 'model__kernel':['linear','rbf'], 
     'model__C': [0.1, 0.5, 0.8, 1, 5],'model__gamma': np.arange(0.2, 1, 0.2)},
    {'model':[KNeighborsClassifier()], 'model__n_neighbors':[5,10,15,20,25]},
    {'model':[DecisionTreeClassifier()], 'model__min_samples_split':[5, 10, 15, 20, 30]}
]

gs = GridSearchCV(model_pl, param_grid=param_grid, 
                  cv=5, return_train_score=True)
gs.fit(X_train, y_train)
score = gs.best_estimator_.score(X_test, y_test)
print('最佳預測模型和參數', gs.best_params_['model'])
print('訓練集的最佳結果', gs.best_score_)
print('測試集的預測結果', score)