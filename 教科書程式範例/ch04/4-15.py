plt.figure(figsize=(10,4))
plt.scatter(X_train, y_train, color='blue', alpha=0.4, label='訓練集')
plt.scatter(X_test, y_test, color='red', alpha=0.4, label='測試集')
plt.xlabel('房間數量')
plt.ylabel('房價')
plt.legend();