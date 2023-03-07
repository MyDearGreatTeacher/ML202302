X_col_cat = ['size','color']
X_cat = X[X_col_cat]
X_cat.style.highlight_null(null_color='yellow')