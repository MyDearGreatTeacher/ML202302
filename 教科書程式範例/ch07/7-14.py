model_pl = make_pipeline(StandardScaler(), 
                         PCA(n_components=2), 
                         KNeighborsClassifier())
model_pl.fit(X_train, y_train)
y_pred = model_pl.predict(X_test)
print('整體正確率:',accuracy_score(y_test, y_pred).round(2))