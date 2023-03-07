fig, axes = plt.subplots(2,3, figsize=(8,5), sharex=True, sharey=True)
scores = []
gammas = [0.001, 0.01, 0.1, 1, 10, 100]
for ax, gamma in zip(axes.ravel(), gammas):
    model_pl = make_pipeline(StandardScaler(), SVC(gamma=gamma))
    model_pl.fit(X_train, y_train)
    plot_decision_boundary(X_train, y_train, model_pl, ax)
    ax.set_title(f'參數gamma={gamma}')
plt.tight_layout()