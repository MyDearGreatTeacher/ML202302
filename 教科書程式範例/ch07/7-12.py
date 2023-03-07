# 原本資料
plt.scatter(x, y)
# 將X_pca轉到原本的資料維度
X_new = pca.inverse_transform(X_pca)
plt.scatter(X_new[:,0], X_new[:,1], c='r', alpha=0.3);