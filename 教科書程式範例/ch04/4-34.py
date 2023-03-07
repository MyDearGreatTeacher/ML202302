errors_train = []
errors_test = []
for order in range(1, 10):
    model_pl_o = make_pipeline(PolynomialFeatures(degree=order), LinearRegression())
    model_pl_o.fit(X_train, y_train)
    y_pred = model_pl_o.predict(X_train)
    errors_train.append(mean_squared_error(y_train, y_pred))
    y_pred = model_pl_o.predict(X_test)
    errors_test.append(mean_squared_error(y_test, y_pred))
    
plt.plot(range(1,10),errors_train, marker='.', ls = '--', label='訓練集')
plt.plot(range(1,10),errors_test, marker='o', label='測試集')
plt.legend();