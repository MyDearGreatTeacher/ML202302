from sklearn.neighbors import KNeighborsClassifier
# 初始物件
model = KNeighborsClassifier()
# 機器學習
model.fit(X_train, y_train)
# 正確率的預測，model.score提供了簡便的正確率輸出方式
model.score(X_test, y_test)