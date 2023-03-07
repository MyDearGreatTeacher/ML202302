print(f'PCA的轉換係數：{pca.components_}')
xy_0 = np.array([x[0],y[0]])
print(f'第一筆原始資料：{xy_0}')
# 進行內積
print(f'自行運算的內積結果：{np.sum(pca.components_ * xy_0)}')
print(f'主成分的第一筆資料：{X_pca[0]}')