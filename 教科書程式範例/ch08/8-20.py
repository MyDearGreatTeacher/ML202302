data_pl = ColumnTransformer([
    ('num_pl', num_pl, ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass']),
    ('cat_pl', cat_pl, ['Sex', 'Embarked'])
])
model_pl_svc = make_pipeline(data_pl, SVC())
model_pl_svc.fit(X_train, y_train)
y_pred = model_pl_svc.predict(X_test)
print('正確率：', accuracy_score(y_test, y_pred).round(2))
print('混亂矩陣')
print(confusion_matrix(y_test, y_pred))