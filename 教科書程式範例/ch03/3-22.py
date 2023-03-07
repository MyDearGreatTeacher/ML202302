data_pl = ColumnTransformer([
    ('num_pl', num_pl, ['price']),
    ('cat_pl', cat_pl, X_col_cat)
], remainder='drop')
pd.DataFrame(data_pl.fit_transform(X))