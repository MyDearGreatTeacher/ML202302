def predict(param):
    model_pl = Pipeline([
        ('preprocess', StandardScaler()),
        ('model', LogisticRegression())
    ])   
    model_pl.set_params(**param)
    model_pl.fit(X_train, y_train)
    return model_pl.score(X_test, y_test)
df_cv = pd.DataFrame(gs.cv_results_, columns=['params','mean_test_score'])
df_cv['model_name'] = df_cv['params'].\
apply(lambda x: x['model'].__class__.__name__)
df_cv['accuracy'] = df_cv['params'].apply(predict)
df_cv = df_cv.set_index('model_name')[['mean_test_score','accuracy']].\
sort_values('mean_test_score', ascending=False)
df_cv.head(5)