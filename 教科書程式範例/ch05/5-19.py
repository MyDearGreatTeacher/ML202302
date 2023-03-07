# 用字典存放初始化的轉換器和預測器
pl = dict()
pl['ss'] = StandardScaler()
pl['regression'] = LinearRegression()

# 訓練集會做標準化的學習和轉換，再進行預測器的學習。
pl['regression'].fit(pl['ss'].fit_transform(X_train), y_train)
# 測試集會做標準化的轉換，和預測器的預測。
y_pred = pl['regression'].predict(pl['ss'].transform(X_test))
print('Mean Squred Error:',mean_squared_error(y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('R2 Score:', r2_score(y_test, y_pred))