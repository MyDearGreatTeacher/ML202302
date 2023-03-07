def predict(param):
    model_pl = Pipeline([
        ('preprocess', StandardScaler()),
        ('model', LogisticRegression())
    ])
    model_pl.set_params(**param)
    model_pl.fit(X_train, y_train)
    return model_pl.score(X_test, y_test)

df_cv = pd.DataFrame(gs.cv_results_)[['params','mean_test_score']].\
sort_values(by = 'mean_test_score', ascending=False)
df_cv_top10 = df_cv.iloc[:10]
# 模型名稱
df_cv_top10['model_name'] = df_cv_top10['params'].\
apply(lambda x: x['model'].__class__.__name__)
# 測試集正確率
df_cv_top10['accuracy'] = df_cv_top10['params'].\
apply(predict)
df_cv_top10 = df_cv_top10.set_index('model_name')[['mean_test_score','accuracy']]
df_cv_top10