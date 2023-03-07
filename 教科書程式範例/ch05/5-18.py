from sklearn.preprocessing import FunctionTransformer
data_pl = ColumnTransformer([
    ('column_sel','passthrough',['LSTAT'])
])
model_pl = make_pipeline(data_pl, 
                         FunctionTransformer(np.log1p),
                         LinearRegression())
model_pl.fit(X_train, y_train)
y_pred = model_pl.predict(X_test)

print('Mean Squred Error:',mean_squared_error(y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('R2 Score:', r2_score(y_test, y_pred))