from sklearn.pipeline import make_pipeline
model_pl_2 = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model_pl_2.fit(X_train, y_train)
y_pred = model_pl_2.predict(X_test)
print('Mean Squred Error:',mean_squared_error(y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('R2 Score:', r2_score(y_test, y_pred))