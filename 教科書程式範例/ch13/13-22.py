model_pl_rf = Pipeline([
    ('preprocess', data_pl),
    ('model', RandomForestClassifier(random_state=42))
])
model_pl_rf.fit(X_train, y_train)
imp = model_pl_rf.named_steps['model'].feature_importances_
feature_names = model_pl_rf.named_steps['preprocess'].\
named_transformers_['cat'].get_feature_names(['sales','salary'])
cols = X_col_num.tolist() + feature_names.tolist()
pd.DataFrame(zip(cols, imp), columns=['欄位', '係數']).\
sort_values(by='係數', ascending=False).head()