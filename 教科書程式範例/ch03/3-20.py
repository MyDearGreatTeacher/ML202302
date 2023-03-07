from sklearn.compose import ColumnTransformer
data_pl = ColumnTransformer([
    ('num_pl', 'drop', X_col_num),
    ('cat_pl', cat_pl, X_col_cat)
])
pd.DataFrame(data_pl.fit_transform(X))