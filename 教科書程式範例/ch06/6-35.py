def plot_decision_boundary(X_test, y_test, model, debug=False):
    points = 500
    x1_max, x2_max = X_test.max()
    x1_min, x2_min = X_test.min()
    X1, X2 = np.meshgrid(np.linspace(x1_min-0.1, x1_max+0.1, points),
                        np.linspace(x2_min-0.1, x2_max+0.1, points))
    x1_label, x2_label = X_test.columns
    fig, ax = plt.subplots()
    X_test.plot(kind='scatter', x=x1_label, y=x2_label, c=y_test, cmap='coolwarm', 
                colorbar=False, figsize=(6,4), s=30, ax=ax)
    grids = np.array(list(zip(X1.ravel(), X2.ravel())))
    ax.contourf(X1, X2, model.predict(grids).reshape(X1.shape), alpha=0.3,
               cmap='coolwarm')
    if debug:
        df_debug = X_test.copy()        
        df_debug['y_test'] = y_test
        y_pred = model.predict(X_test)
        df_debug['y_pred'] = y_pred
        df_debug = df_debug[y_pred != y_test]
        df_debug.plot(kind='scatter', x=x1_label, y=x2_label, 
                      s=50,  color='none', edgecolor='y', ax=ax)
        for i in df_debug.index:
            ax.text(s=df_debug.loc[i,'y_test'], x=df_debug.loc[i, x1_label]+0.01,
                       y=df_debug.loc[i, x2_label]-0.05)