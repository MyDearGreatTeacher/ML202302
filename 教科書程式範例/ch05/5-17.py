fig, axes = plt.subplots(1, 2, figsize=(8,3))
df['LSTAT'].hist(alpha=0.4, bins=30, ax=axes[0])
axes[0].set_title('原始')
# 對'LSTAT'欄位進行log轉換
np.log1p(df['LSTAT']).hist(alpha=0.4, bins=30, ax=axes[1])
axes[1].set_title('進行log轉換');