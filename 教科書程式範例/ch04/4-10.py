plt.figure(figsize=(8, 6))
corr[np.abs(corr) < 0.6] = 0
sns.heatmap(corr, annot=True, cmap='coolwarm');