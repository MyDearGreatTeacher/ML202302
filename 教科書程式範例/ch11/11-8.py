model_pl = make_pipeline(StandardScaler(), SVC())
gs = GridSearchCV(model_pl, param_grid=param_grid, 
                  cv=10, return_train_score=True)
# 進行網格搜尋學習
gs.fit(X_train, y_train)