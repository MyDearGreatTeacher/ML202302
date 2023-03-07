from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
fig, axes = plt.subplots(2,3, figsize=(8,5), sharex=True, sharey=True)
scores = []
Cs = [0.001, 0.01, 0.1, 1, 10, 100]
for ax, c in zip(axes.ravel(), Cs):
    model_pl = make_pipeline(StandardScaler(), SVC(C=c))
    model_pl.fit(X_train, y_train)
    plot_decision_boundary(X_train, y_train, model_pl, ax)
    ax.set_title(f'參數C={c}')
plt.tight_layout()