models = [LogisticRegression(), SVC(), 
          KNeighborsClassifier(), DecisionTreeClassifier(max_depth=10)]
scores = {}
for model in models:
    model_pl = make_pipeline(StandardScaler(), model)
    score = cross_val_score(model_pl, X_train, y_train, scoring='accuracy', cv=10)
    scores[model.__class__.__name__] = score.mean()
pd.Series(scores).sort_values(ascending=False)