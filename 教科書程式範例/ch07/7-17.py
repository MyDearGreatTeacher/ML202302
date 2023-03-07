model_pl = make_pipeline(StandardScaler(), 
                        SelectKBest(f_classif, 2),
                        KNeighborsClassifier())
model_pl.fit(X_train, y_train)
y_pred = model_pl.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('整體正確率:',accuracy_score(y_test, y_pred).round(2))